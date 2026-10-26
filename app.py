import gradio as gr
import sys
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from generate import generate_answer


CSS = """
:root {
    --forest: #0A3323;
    --moss: #839958;
    --cream: #F7F4D5;
    --rose: #D3968C;
    --midnight: #105666;
}

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    background: var(--cream) !important;
}

body {
    background: var(--cream) !important;
}

#hero {
    padding: 38px 36px;
    border-radius: 22px;
    background: var(--forest);
    margin-bottom: 24px;
    border: 1px solid var(--moss);
}

#hero h1 {
    font-size: 42px;
    line-height: 1.15;
    letter-spacing: -1.5px;
    margin: 12px 0;
    color: var(--cream);
}

#hero p {
    color: #DCE6CF;
    font-size: 15px;
    line-height: 1.8;
    max-width: 720px;
}

#eyebrow {
    color: var(--moss);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.5px;
}

h2, h3 {
    color: var(--forest) !important;
}

#question-box textarea {
    font-size: 16px !important;
    background: #FFFEF5 !important;
    color: var(--forest) !important;
}

#ask-button {
    min-height: 48px;
    border-radius: 12px;
    background: var(--forest) !important;
    border: 1px solid var(--forest) !important;
    color: var(--cream) !important;
    font-weight: 700;
}

#ask-button:hover {
    background: var(--midnight) !important;
}

#suggestions button {
    border-radius: 10px;
    background: #E8EACF !important;
    color: var(--forest) !important;
    border: 1px solid #C5CDA7 !important;
}

#suggestions button:hover {
    background: var(--moss) !important;
    color: white !important;
}

#answer-panel {
    font-size: 16px !important;
    line-height: 1.85 !important;
    padding: 24px !important;
    border: 1px solid #D9DFC4 !important;
    border-left: 5px solid #839958 !important;
    border-radius: 16px !important;
    background: #FFFEF5 !important;
    color: #0A3323 !important;
}

/* Force readable text throughout the answer panel */
#answer-panel,
#answer-panel p,
#answer-panel li,
#answer-panel strong,
#answer-panel em,
#answer-panel h1,
#answer-panel h2,
#answer-panel h3 {
    color: #0A3323 !important;
}

#source-panel {
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 16px 24px !important;
    border: 1px solid #C1D5D7 !important;
    border-left: 5px solid #105666 !important;
    border-radius: 16px !important;
    background: #EAF2EF !important;
    color: #0A3323 !important;
}

#source-panel,
#source-panel p,
#source-panel li,
#source-panel strong {
    color: #0A3323 !important;
}

button {
    transition: all 0.2s ease !important;
}

footer {
    display: none !important;
}
/* Readable response timer */
#timer-panel {
    background: #0A3323 !important;
    color: #F7F4D5 !important;
    border: 2px solid #D3968C !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    margin-top: 16px !important;
    text-align: center !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

#timer-panel p,
#timer-panel span {
    color: #F7F4D5 !important;
}

/* Lecture evidence */
#lecture-panel {
    background: #E9EDDA !important;
    color: #0A3323 !important;
    border-left: 5px solid #839958 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

#lecture-panel p,
#lecture-panel li,
#lecture-panel strong {
    color: #0A3323 !important;
}

/* Evidence limitations */
#limitations-panel {
    background: #F8E9E4 !important;
    color: #0A3323 !important;
    border-left: 5px solid #D3968C !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

#limitations-panel p,
#limitations-panel li,
#limitations-panel strong {
    color: #0A3323 !important;
}

/* Source panel */
#source-panel {
    background: #E4EEEE !important;
    color: #105666 !important;
    border-left: 5px solid #105666 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

#source-panel p,
#source-panel li,
#source-panel strong {
    color: #105666 !important;
}
"""
def ask_algolearn(question):
    if not question or not question.strip():
        return (
            "Enter a question to begin.",
            "",
            "",
            "",
            "",
        )

    import time
    start_time = time.perf_counter()

    try:
        raw_answer, results = generate_answer(question.strip())

        # Split the generated response into three sections.
        headings = [
            "WHAT THE LECTURE STATES:",
            "WHAT THE EXCERPTS DO NOT ESTABLISH:",
        ]

        answer_text = raw_answer
        lecture_text = ""
        limitations_text = ""

        if headings[0] in raw_answer:
            answer_text, remaining = raw_answer.split(
                headings[0], 1
            )

            if headings[1] in remaining:
                lecture_text, limitations_text = remaining.split(
                    headings[1], 1
                )
            else:
                lecture_text = remaining
        elif headings[1] in raw_answer:
            answer_text, limitations_text = raw_answer.split(
                headings[1], 1
            )

        # Remove the leading ANSWER label.
        answer_text = answer_text.strip()
        if answer_text.upper().startswith("ANSWER:"):
            answer_text = answer_text[len("ANSWER:"):].strip()

        lecture_text = lecture_text.strip()
        limitations_text = limitations_text.strip()

        # Remove duplicate source references.
        seen = set()
        source_lines = []

        for item in results:
            key = (item["source"], item["page"])

            if key not in seen:
                seen.add(key)
                source_lines.append(
                    f"- **{item['source']}** — Page {item['page']}"
                )

        source_text = "\n\n".join(source_lines)
        elapsed = time.perf_counter() - start_time

        timer_text = (
            f"⏱️  RESPONSE TIME  ·  {elapsed:.2f} seconds"
        )

        return (
            answer_text,
            lecture_text,
            limitations_text,
            source_text,
            timer_text,
        )

    except Exception as error:
        return (
            f"Something went wrong: {error}",
            "",
            "",
            "",
            f"Request failed: {error}",
        )

with gr.Blocks(
    title="AlgoLearn | Evidence-first DSA Tutor",
    theme=gr.themes.Base(
      primary_hue="green",
      secondary_hue="orange",
      neutral_hue="stone",
    ),
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div id="hero">
            <div id="eyebrow">YOUR ALGORITHM LEARNING SPACE</div>
            <h1>Understand the why.<br>Not just the answer.</h1>
            <p>
                An evidence-first DSA tutor grounded in MIT's
                Design and Analysis of Algorithms lecture material.
                Ask a question, inspect the sources, and learn with confidence.
            </p>
        </div>
        """
    )

    gr.Markdown("## Ask AlgoLearn")
    gr.Markdown(
        "Start with a question. Every answer comes with the retrieved "
        "lecture references used to generate it."
    )

    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. Why can Quicksort take Θ(n²) time?",
        lines=2,
        elem_id="question-box",
    )

    with gr.Row():
        submit = gr.Button(
            "Explain concept →",
            variant="primary",
            elem_id="ask-button",
            scale=3,
        )
        clear = gr.Button("Clear", scale=1)

    gr.Markdown("### Answer")
    answer = gr.Markdown(
        "Ask a question to begin.",
        elem_id="answer-panel",
    )

    gr.Markdown("### What the Lecture States")
    lecture_states = gr.Markdown(
        "Relevant statements from the lecture will appear here.",
        elem_id="lecture-panel",
    )

    gr.Markdown("### What the Excerpts Do Not Establish")
    limitations = gr.Markdown(
        "Evidence limitations will appear here.",
        elem_id="limitations-panel",
    )

    gr.Markdown("### Sources")
    sources = gr.Markdown(
        "MIT lecture references will appear here.",
        elem_id="source-panel",
    )

    timer_display = gr.Markdown(
        "⏱️ Response time will appear here.",
        elem_id="timer-panel",
    )

    gr.Markdown(
        """
        ---
        <div style="text-align:center; opacity:0.75; font-size:12px">
        ALGOLEARN · SOURCE-GROUNDED LEARNING · MIT 6.046J
        <br>
        Answers may be incomplete when the indexed lecture excerpts
        do not contain enough evidence.
        </div>
        """
    )

    submit.click(
        fn=ask_algolearn,
        inputs=question,
        outputs=[
            answer,
            lecture_states,
            limitations,
            sources,
            timer_display,
        ],
        show_progress="minimal",
    )

    question.submit(
        fn=ask_algolearn,
        inputs=question,
        outputs=[
            answer,
            lecture_states,
            limitations,
            sources,
            timer_display,
        ],
        show_progress="minimal",
    )

    clear.click(
        fn=lambda: (
            "Ask a question to begin.",
            "Relevant lecture statements will appear here.",
            "Evidence limitations will appear here.",
            "MIT lecture references will appear here.",
            "⏱️ Response time will appear here.",
            "",
        ),
        inputs=None,
        outputs=[
            answer,
            lecture_states,
            limitations,
            sources,
            timer_display,
            question,
        ],
    )

if __name__ == "__main__":
    demo.launch()