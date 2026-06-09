"""
search.py
---------
Stretch features layered on top of the base semantic retrieval:

  1. Metadata filtering  — detect a professor name in the query and restrict
     retrieval to that professor's reviews (fixes the "name is a weak signal"
     failure where a question about one professor pulls another's reviews).

  2. Hybrid search       — combine semantic similarity (all-MiniLM-L6-v2 +
     ChromaDB) with BM25 keyword scoring, so an exact token like a professor's
     surname is weighted heavily instead of being washed out by topic words.

Functions:
    detect_professor(query)                  -> str | None
    retrieve_semantic(query, k, professor)   -> list[dict]   (base + optional filter)
    retrieve_hybrid(query, k, alpha, professor) -> list[dict]
    retrieve_smart(query, k)                 -> list[dict]   (auto filter + hybrid)
"""

import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import get_chunks
from embed import get_collection, EMBED_MODEL

# ── Lazy singletons ───────────────────────────────────────────────────────────

_model      = None   # sentence-transformers embedder
_collection = None   # ChromaDB collection
_chunks     = None   # flat list of chunk dicts (same order as ids chunk_0..n)
_bm25       = None   # BM25 index over chunk texts
_professors = None   # set of professor names present in the corpus


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


def _get_chunks():
    """Chunk i here corresponds to ChromaDB id 'chunk_i' (same get_chunks() order)."""
    global _chunks
    if _chunks is None:
        _chunks = get_chunks()
    return _chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25():
    global _bm25
    if _bm25 is None:
        corpus = [_tokenize(c["text"]) for c in _get_chunks()]
        _bm25 = BM25Okapi(corpus)
    return _bm25


def _get_professors():
    global _professors
    if _professors is None:
        _professors = sorted({c["professor"] for c in _get_chunks() if c["professor"]})
    return _professors


# ── Metadata filtering ────────────────────────────────────────────────────────

def detect_professor(query: str) -> str | None:
    """
    Return a professor's full name if it (or their surname) appears in the query.

    Matches the full name ("craig kapp") or the surname as a whole word ("kapp"),
    so "What are Kapp's exams like?" resolves to "Craig Kapp".
    """
    q_words = set(_tokenize(query))
    q_lower = query.lower()
    for name in _get_professors():
        parts = name.lower().split()
        if name.lower() in q_lower:
            return name
        if len(parts) > 1 and parts[-1] in q_words:
            return name
    return None


# ── Semantic retrieval (with optional metadata filter) ────────────────────────

def retrieve_semantic(query: str, k: int = 5, professor: str | None = None) -> list[dict]:
    """Top-k semantic search, optionally restricted to one professor via ChromaDB `where`."""
    model      = _get_model()
    collection = _get_collection()

    query_vec = model.encode(query, convert_to_list=True)
    kwargs = {
        "query_embeddings": [query_vec],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if professor:
        kwargs["where"] = {"professor": professor}

    results = collection.query(**kwargs)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append({
            "text":      doc,
            "professor": meta.get("professor", "Unknown"),
            "source":    meta.get("source", ""),
            "rmp_url":   meta.get("rmp_url", ""),
            "distance":  round(dist, 4),
        })
    return out


# ── Hybrid search (semantic + BM25) ───────────────────────────────────────────

def _minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def retrieve_hybrid(
    query: str, k: int = 5, alpha: float = 0.5, professor: str | None = None
) -> list[dict]:
    """
    Combine semantic similarity and BM25 keyword scores.

        combined = alpha * semantic_norm + (1 - alpha) * bm25_norm

    Both component scores are min-max normalized to [0, 1] over the candidate set
    so they're comparable. alpha=1.0 is pure semantic, alpha=0.0 is pure BM25.
    `professor`, if given, restricts the candidate set to that professor's chunks.
    """
    chunks     = _get_chunks()
    model      = _get_model()
    collection = _get_collection()
    bm25       = _get_bm25()

    # Candidate indices (all chunks, or only the named professor's)
    candidates = [
        i for i, c in enumerate(chunks)
        if professor is None or c["professor"] == professor
    ]

    # Semantic similarity for every candidate: pull distances for all chunks at once
    query_vec = model.encode(query, convert_to_list=True)
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=len(chunks),
        include=["distances"],
    )
    # ChromaDB ids are "chunk_i"; map id -> distance, then index -> similarity
    id_to_dist = dict(zip(res["ids"][0], res["distances"][0]))
    sem_sim = [1.0 - id_to_dist.get(f"chunk_{i}", 1.0) for i in candidates]  # cosine sim

    # BM25 score for every candidate
    bm25_all = bm25.get_scores(_tokenize(query))
    bm25_cand = [bm25_all[i] for i in candidates]

    sem_norm  = _minmax(sem_sim)
    bm25_norm = _minmax(bm25_cand)

    scored = []
    for pos, i in enumerate(candidates):
        combined = alpha * sem_norm[pos] + (1 - alpha) * bm25_norm[pos]
        c = chunks[i]
        scored.append({
            "text":        c["text"],
            "professor":   c["professor"],
            "source":      c["source"],
            "rmp_url":     c["rmp_url"],
            "distance":    round(id_to_dist.get(f"chunk_{i}", 1.0), 4),
            "bm25":        round(bm25_cand[pos], 3),
            "hybrid_score": round(combined, 4),
        })

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored[:k]


# ── Smart retrieval: auto professor filter + hybrid ───────────────────────────

def retrieve_smart(query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
    """Detect a professor in the query, then run hybrid search (filtered if found)."""
    professor = detect_professor(query)
    return retrieve_hybrid(query, k=k, alpha=alpha, professor=professor)


# ── CLI quick check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What do students say about Craig Kapp's exams and grading?"
    prof = detect_professor(q)
    print(f"\nQuery: {q}")
    print(f"Detected professor: {prof or '(none)'}\n")
    print("── Smart (auto-filter + hybrid) ──")
    for i, c in enumerate(retrieve_smart(q), 1):
        print(f"  [{i}] hybrid={c['hybrid_score']} (sem_dist={c['distance']}, bm25={c['bm25']}) "
              f"| {c['professor']} | {c['source']}")
        print(f"      {c['text'][:110]}...")