"""
query.py
--------
Retrieval and grounded generation.

    retrieve(query, k=5)  → list of chunk dicts with distance scores
    ask(question)         → {"answer": str, "sources": list[str]}
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from embed import get_collection, EMBED_MODEL

load_dotenv()

_model      = None   # lazy-loaded embedder
_collection = None   # lazy-loaded ChromaDB collection


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed query and return top-k chunks from ChromaDB.

    Returns a list of dicts:
        {text, professor, source, rmp_url, distance}
    """
    model      = _get_model()
    collection = _get_collection()

    query_vec = model.encode(query, convert_to_list=True)
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":      doc,
            "professor": meta.get("professor", "Unknown"),
            "source":    meta.get("source", ""),
            "rmp_url":   meta.get("rmp_url", ""),
            "distance":  round(dist, 4),
        })

    return chunks


# ── Grounded generation ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about NYU Computer Science professors based ONLY on student reviews provided to you as context.

Rules you must follow:
1. Answer ONLY from the context below. Do not use outside knowledge.
2. At the end of your answer, cite the source files you used, like: Sources: craig_kapp.txt, ben_goldberg.txt
3. If the context does not contain enough information to answer the question confidently, respond with exactly: "I don't have enough information to answer that based on the available reviews."
4. Do not speculate or fill gaps with assumptions.
5. Be concise and direct."""


def ask(question: str, k: int = 5, mode: str = "semantic") -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer via Groq.

    Args:
        mode: "semantic" (base top-k vector search) or "smart" (stretch:
              auto professor metadata-filter + hybrid BM25/semantic search).

    Returns:
        {
            "answer":  str,
            "sources": list[str],   # deduplicated source filenames
            "chunks":  list[dict],  # raw retrieved chunks (for transparency)
        }
    """
    if mode == "smart":
        from search import retrieve_smart
        chunks = retrieve_smart(question, k=k)
    else:
        chunks = retrieve(question, k=k)

    # Build context block from retrieved chunks
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {c['source']} | Professor: {c['professor']}\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    user_message = f"""Context (student reviews):
{context}

Question: {question}"""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment / .env file.")

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temp for factual grounding
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()

    # Deduplicate source filenames preserving order
    seen    = set()
    sources = []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append(c["source"])

    return {"answer": answer, "sources": sources, "chunks": chunks}


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What do students say about Craig Kapp's exams and grading?"

    print(f"\nQuery: {q}\n{'─'*60}")
    result = ask(q)
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")
    print(f"\n── Retrieved chunks ──")
    for c in result["chunks"]:
        print(f"  [{c['distance']}] {c['professor']} | {c['source']}")
        print(f"  {c['text'][:120]}...\n")
