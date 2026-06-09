"""
embed.py
--------
Embeds all chunks from ingest.py and stores them in a local ChromaDB collection.

Usage:
    python embed.py              # builds / rebuilds the vector store
    from embed import get_collection  # import into query.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from ingest import get_chunks

COLLECTION_NAME = "nyu_rmp_reviews"
CHROMA_PATH     = "./chroma_db"
EMBED_MODEL     = "all-MiniLM-L6-v2"


def build_vectorstore(folder_path=None):
    """Embed all chunks and persist them to ChromaDB."""
    print("Loading chunks...")
    chunks = get_chunks(folder_path)
    if not chunks:
        raise RuntimeError("No chunks found — check your docs/ folder.")

    print(f"  {len(chunks)} chunks loaded from {len(set(c['source'] for c in chunks))} files")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks (this may take a moment)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate so re-runs don't duplicate
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Dropped existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "professor": c["professor"],
            "source":    c["source"],
            "rmp_url":   c["rmp_url"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n✓ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")
    return collection


def get_collection():
    """Return the existing ChromaDB collection (must have run build_vectorstore first)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    build_vectorstore()
