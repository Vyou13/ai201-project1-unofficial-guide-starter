"""
app.py
------
Gradio interface for the NYU CS Professor Review RAG system.

Usage:
    python app.py
"""

import gradio as gr
from query import ask


def answer_question(question: str, mode: str = "smart"):
    """Called by Gradio on submit. Returns (answer_text, sources_text)."""
    if not question.strip():
        return "Please enter a question.", ""

    # mode: "Smart (filter + hybrid)" → smart ; "Semantic only" → semantic
    backend = "smart" if mode.lower().startswith("smart") else "semantic"
    result = ask(question, mode=backend)

    answer  = result["answer"]
    sources = result["sources"]
    chunks  = result["chunks"]

    # Format sources with RMP URLs
    source_lines = []
    seen = set()
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            line = f"• {c['professor']} — {c['source']}"
            if c["rmp_url"]:
                line += f"\n  {c['rmp_url']}"
            source_lines.append(line)

    sources_text = "\n".join(source_lines) if source_lines else "No sources found."

    return answer, sources_text


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="NYU CS Professor Reviews") as demo:
    gr.Markdown(
        """
        # 🎓 NYU CS Professor Review Search
        Ask questions about NYU Computer Science professors based on real Rate My Professors reviews.
        Answers are grounded in student reviews only — no hallucination.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What do students say about Craig Kapp's exams?",
                lines=2,
            )
            mode_radio = gr.Radio(
                choices=["Smart (filter + hybrid)", "Semantic only"],
                value="Smart (filter + hybrid)",
                label="Search mode",
                info="Smart auto-detects a professor in the query and blends BM25 keyword "
                     "search with semantic search. Switch to Semantic only to see the "
                     "baseline (and the Craig Kapp retrieval failure).",
            )
            submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
            )
        with gr.Column(scale=2):
            sources_box = gr.Textbox(
                label="Sources",
                lines=8,
                interactive=False,
            )

    gr.Examples(
        examples=[
            ["What do students say about Craig Kapp's exams and grading?"],
            ["What is the main complaint students have about Michael Tao's intro CS class?"],
            ["Is Alan Siegel a good professor for algorithms?"],
            ["What do students say about Ben Goldberg's availability and responsiveness?"],
            ["Which professor would be best for a beginner with no programming experience?"],
            ["What do students think about Patrick Cousot's teaching style?"],
        ],
        inputs=question_box,
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, mode_radio],
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        fn=answer_question,
        inputs=[question_box, mode_radio],
        outputs=[answer_box, sources_box],
    )

if __name__ == "__main__":
    demo.launch()
