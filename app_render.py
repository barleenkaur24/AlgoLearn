import json
import os
import re
from pathlib import Path

import gradio as gr
from google import genai

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"
MODEL_NAME = "gemini-2.5-flash"

STOPWORDS = {
    "what", "why", "how", "when", "where", "which", "who",
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "of", "to", "in", "on", "for", "with", "by",
    "it", "this", "that", "be", "can", "does", "do",
    "explain", "tell", "me", "about", "please"
}


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Lecture chunks not found: {CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("The lecture chunk file is empty.")

    return chunks


CHUNKS = load_chunks()


def retrieve(question, top_k=4):
    query_words = {
        word for word in re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
        if word not in STOPWORDS and len(word) > 1
    }

    if not query_words:
        query_words = set(question.lower().split())

    scored = []

    for chunk in CHUNKS:
        text = chunk.get("text", "")
        words = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        )

        if not words:
            continue

        overlap = len(query_words & words)
        score = overlap / (len(query_words) ** 0.5)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def answer_question(question):
    question = question.strip()

    if not question:
        return "Please enter a DSA question."

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return (
            "The hosted AI service is not configured yet. "
            "Please configure GEMINI_API_KEY in the deployment "
            "environment variables."
        )

    results = retrieve(question)

    if not results:
        return (
            "I couldn't find a sufficiently relevant passage in "
            "the indexed lecture. Try using terms from the "
            "Quicksort lecture."
        )

    context_parts = []

    for chunk in results:
        source = chunk.get("source", "MIT Lecture 6")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source: {source}, page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are AlgoLearn, a DSA tutor specializing in Quicksort.

Answer the student's question using the lecture excerpts below.

Rules:
- Ground factual claims in the supplied excerpts.
- Do not invent claims or proofs that the excerpts do not establish.
- Distinguish basic, median-based, randomized, and Paranoid Quicksort.
- Cite relevant claims using [MIT Lecture 6, p. X] when the page
  number is available.
- If the excerpts do not establish a claim, state that clearly.
- Explain concepts clearly for a university student.
- Do not claim that the retrieved excerpts prove something they
  merely mention.

Use these sections:
1. ANSWER
2. WHAT THE LECTURE STATES
3. WHAT THE EXCERPTS DO NOT ESTABLISH

LECTURE EXCERPTS:
{context}

STUDENT QUESTION:
{question}
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 700,
            },
        )

        generated_text = response.text

        if not generated_text:
            return "The AI service returned an empty response. Please retry."

        return generated_text

    except Exception as error:
        print(f"Generation error: {error}")
        return (
            "Sorry, the AI service couldn't generate an answer. "
            "Please retry shortly. If the problem continues, "
            "check the deployment logs and API configuration."
        )


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask AlgoLearn",
        placeholder="Explain randomized Quicksort...",
        lines=3,
    ),
    outputs=gr.Markdown(label="Answer"),
    title="AlgoLearn — Your DSA Tutor",
    description=(
        "Ask questions about Quicksort. AlgoLearn retrieves "
        "relevant passages from the MIT lecture and uses AI "
        "to explain them with source references."
    ),
    examples=[
        "How does randomized Quicksort work?",
        "What is the difference between randomized and basic Quicksort?",
        "What does the lecture establish about expected running time?",
    ],
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)