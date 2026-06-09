"""
test_pipeline.py
----------------
Verifies all three pipeline stages against your 5 evaluation questions.

Stages tested:
  1. Retrieval  — checks top-k chunks are returned and distance scores are reasonable
  2. Generation — calls ask() and prints expected vs. actual for manual judgment
  3. Out-of-scope — confirms the system refuses to answer unknown questions

Usage:
    python test_pipeline.py
"""

from query import retrieve, ask

SEPARATOR = "=" * 70

# ── Evaluation questions from planning.md ─────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "What do students say about Craig Kapp's exams and grading?",
        "expected": (
            "Tests and quizzes are fair with no tricks, there's extra credit, "
            "and it's easy to get a good grade if you do the work. "
            "Difficulty rated around 2-3/5."
        ),
    },
    {
        "id": 2,
        "question": "What is the main complaint students have about Michael Tao's intro CS class?",
        "expected": (
            "The exams are extremely long, confusing, and much harder than the lectures. "
            "Students report not finishing exams and that half the class fails the second midterm."
        ),
    },
    {
        "id": 3,
        "question": "Is Alan Siegel a good professor for algorithms?",
        "expected": (
            "No — 2.7/5 rating, only 35% would take again. "
            "Students describe him as a tough grader; some say he's caring but the class is very hard."
        ),
    },
    {
        "id": 4,
        "question": "What do students say about Ben Goldberg's availability and responsiveness?",
        "expected": (
            "Responds quickly, very nice about extensions, accessible outside class; 4.7/5 overall."
        ),
    },
    {
        "id": 5,
        "question": "Which professor would be best for a beginner with no programming experience?",
        "expected": (
            "Craig Kapp or Candido Cabo. Kapp is 4.8/5 — great intro. "
            "Cabo is described as great at explaining concepts to beginners with generous curves."
        ),
    },
]

OUT_OF_SCOPE_QUESTION = "What is the best pizza place near NYU?"


# ── Stage 1: Retrieval test ───────────────────────────────────────────────────

def test_retrieval():
    print(f"\n{SEPARATOR}")
    print("STAGE 1 — RETRIEVAL TEST")
    print(f"{SEPARATOR}")

    # Run 3 of the 5 eval questions through retrieve()
    sample_qs = EVAL_QUESTIONS[:3]
    all_passed = True

    for item in sample_qs:
        q = item["question"]
        print(f"\nQ{item['id']}: {q}")
        chunks = retrieve(q, k=5)

        print(f"  Top {len(chunks)} chunks returned:")
        for i, c in enumerate(chunks, 1):
            flag = "✓" if c["distance"] < 0.8 else "⚠"
            print(f"  {flag} [{i}] dist={c['distance']} | {c['professor']} | {c['source']}")
            print(f"       {c['text'][:100]}...")

        top_dist = chunks[0]["distance"] if chunks else 1.0
        if top_dist >= 0.8:
            print(f"  ⚠ WARNING: top distance {top_dist} is high — retrieval may be off-topic")
            all_passed = False
        else:
            print(f"  ✓ Top distance {top_dist} looks good (< 0.8)")

    print(f"\nRetrieval stage: {'✓ PASSED' if all_passed else '⚠ CHECK WARNINGS ABOVE'}")


# ── Stage 2: Generation test (all 5 eval questions) ──────────────────────────

def test_generation():
    print(f"\n{SEPARATOR}")
    print("STAGE 2 — GENERATION TEST (all 5 eval questions)")
    print(f"{SEPARATOR}")

    for item in EVAL_QUESTIONS:
        print(f"\n{'─'*60}")
        print(f"Q{item['id']}: {item['question']}")
        print(f"\nExpected:\n  {item['expected']}")

        result = ask(item["question"])

        print(f"\nActual:\n  {result['answer']}")
        print(f"\nSources cited: {', '.join(result['sources'])}")

        # Flag if the system refused when it shouldn't have
        refused = "don't have enough information" in result["answer"].lower()
        if refused:
            print("  ⚠ System refused — check if relevant chunks were retrieved")
        else:
            print("  ✓ System produced an answer")

        print("\n[Manual judgment required: accurate / partially accurate / inaccurate]")


# ── Stage 3: Out-of-scope refusal test ───────────────────────────────────────

def test_out_of_scope():
    print(f"\n{SEPARATOR}")
    print("STAGE 3 — OUT-OF-SCOPE REFUSAL TEST")
    print(f"{SEPARATOR}")

    q = OUT_OF_SCOPE_QUESTION
    print(f"\nQuery: {q}")
    result = ask(q)
    print(f"\nSystem response:\n  {result['answer']}")

    refused = "don't have enough information" in result["answer"].lower()
    if refused:
        print("\n✓ PASSED — system correctly refused to answer out-of-scope question")
    else:
        print("\n⚠ FAILED — system should have refused but produced an answer")
        print("  Review the grounding prompt in query.py")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nNYU Professor RAG — Pipeline Test Suite")
    print(SEPARATOR)

    try:
        test_retrieval()
    except Exception as e:
        print(f"\n✗ Retrieval stage failed with error: {e}")
        print("  Make sure you've run `python embed.py` first.")
        raise SystemExit(1)

    try:
        test_generation()
    except Exception as e:
        print(f"\n✗ Generation stage failed with error: {e}")
        raise SystemExit(1)

    try:
        test_out_of_scope()
    except Exception as e:
        print(f"\n✗ Out-of-scope test failed with error: {e}")
        raise SystemExit(1)

    print(f"\n{SEPARATOR}")
    print("All stages complete. Review the output above for accuracy judgments.")
    print(SEPARATOR)
