"""
ingest.py
---------
Loads NYU CS professor review files from a folder, cleans them,
and chunks them into retrievable units with metadata attached.

Usage:
    python ingest.py <folder>      # prints sanity check output
    from ingest import get_chunks  # import into embed.py
"""

import os


def parse_header(text):
    """Extract professor name and RMP URL from the # comment header."""
    professor, rmp_url = "", ""
    for line in text.splitlines():
        if line.startswith("# Professor:"):
            professor = line.replace("# Professor:", "").strip()
        elif line.startswith("# RMP URL:"):
            rmp_url = line.replace("# RMP URL:", "").strip()
    return {"professor": professor, "rmp_url": rmp_url}


def clean_text(text):
    """Strip the # header block; return only the review content."""
    lines = text.splitlines()
    content_start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            content_start = i
            break
    return "\n".join(lines[content_start:]).strip()


def chunk_text(text, chunk_size=400, overlap=50, tolerance=60):
    """
    Split text into chunks.

    Strategy:
      1. Split on blank lines (\\n\\n) — each RMP review is one block.
      2. Keep any block that fits within chunk_size + tolerance as a single
         chunk. The tolerance (default 60 chars) prevents splitting a review
         that is only slightly over the limit into an awkward fragment.
      3. Only apply character-based splitting with overlap for genuinely long
         blocks (e.g. a review that runs 600+ characters). Break at word
         boundaries to avoid cutting mid-word.

    Args:
        text:       cleaned review text (header removed)
        chunk_size: target max characters per chunk
        overlap:    overlap between character-split chunks
        tolerance:  how many chars over chunk_size a block can be before
                    we split it. Prevents 401-char reviews being split into
                    a 399-char chunk + a 52-char orphan fragment.
    """
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    hard_limit = chunk_size + tolerance  # e.g. 460 chars

    chunks = []
    for block in raw_blocks:
        if len(block) <= hard_limit:
            # Block fits — keep whole (one complete review per chunk)
            chunks.append(block)
        else:
            # Genuinely long block — split at word boundaries with overlap
            start = 0
            while start < len(block):
                end = start + chunk_size
                if end < len(block):
                    boundary = block.rfind(" ", start, end)
                    if boundary > start:
                        end = boundary
                chunk = block[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                advance = (end - start) - overlap
                if advance <= 0:
                    advance = chunk_size - overlap
                start += advance

    return chunks


def load_documents(folder_path):
    """
    Read all .txt files in folder_path and return a flat list of chunk dicts.

    Each dict:
        {
            "text":      chunk text (one complete review in most cases),
            "professor": professor name from file header,
            "source":    filename (e.g. "craig_kapp_txt.txt"),
            "rmp_url":   RMP URL from file header,
        }
    """
    all_chunks = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    if not txt_files:
        print(f"Warning: no .txt files found in {folder_path}")
        return []

    for filename in sorted(txt_files):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        meta = parse_header(raw_text)
        content = clean_text(raw_text)
        chunks = chunk_text(content)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "professor": meta["professor"],
                "source": filename,
                "rmp_url": meta["rmp_url"],
            })

    return all_chunks


DEFAULT_DOCS_PATHS = ["docs", "documents/docs", "documents"]


def resolve_folder(folder_path=None):
    """Return an existing documents folder, or a default candidate if none is specified."""
    if folder_path:
        return folder_path
    for candidate in DEFAULT_DOCS_PATHS:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_DOCS_PATHS[0]


def get_chunks(folder_path=None):
    """Convenience wrapper for embed.py."""
    folder = resolve_folder(folder_path)
    return load_documents(folder)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else resolve_folder()
    if not os.path.exists(folder):
        print(
            f"Folder '{folder}' not found. Usage: python ingest.py <folder>\n"
            f"Try one of: {', '.join(DEFAULT_DOCS_PATHS)}"
        )
        sys.exit(1)

    chunks = load_documents(folder)

    print(f"\n{'='*60}")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Files loaded : {len(set(c['source'] for c in chunks))}")
    print(f"{'='*60}\n")

    print("── First 5 chunks ──\n")
    for i, c in enumerate(chunks[:5]):
        print(f"[Chunk {i+1}] {c['professor']} | {c['source']} | {len(c['text'])} chars")
        print(f"  {c['text']}\n")

    lengths = [len(c["text"]) for c in chunks]
    print(f"── Length stats ──")
    print(f"  Min {min(lengths)} | Max {max(lengths)} | Avg {sum(lengths)//len(lengths)} chars")

    fragments = [c for c in chunks if len(c["text"]) < 80]
    if fragments:
        print(f"\n  ⚠ {len(fragments)} short chunk(s) under 80 chars — may be fragments:")
        for f in fragments:
            print(f"    [{f['professor']}] {repr(f['text'][:80])}")
    else:
        print(f"  ✓ No suspicious fragments found")

    print(f"\n── Professor coverage ──\n")
    seen = set()
    for c in chunks:
        if c["professor"] not in seen:
            seen.add(c["professor"])
            count = sum(1 for x in chunks if x["professor"] == c["professor"])
            print(f"  ✓ {c['professor']:25s} {count} chunks")
