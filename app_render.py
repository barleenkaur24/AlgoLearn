import json
import os
import re
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import time

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"
MODEL_NAME = "gemini-2.5-flash-lite"
STOPWORDS = {
    "what", "why", "how", "when", "where", "which", "who",
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "of", "to", "in", "on", "for", "with", "by",
    "it", "this", "that", "be", "can", "does", "do",
    "explain", "tell", "me", "about", "please"
}


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Lecture chunks not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

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
        words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

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
        raise RuntimeError("GEMINI_API_KEY is missing from the deployment environment.")

    results = retrieve(question)

    if not results:
        return (
            "I couldn't find a sufficiently relevant passage in the indexed lecture. "
            "Try using terms from the Quicksort lecture."
        )

    context_parts = []

    for chunk in results:
        source = chunk.get("source", "MIT Lecture 6")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source: {source}, page {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are AlgoLearn, a DSA tutor specializing in Quicksort.

Answer the student's question using only the lecture excerpts below.

Rules:
- Ground factual claims in the supplied excerpts.
- Do not invent claims or proofs that the excerpts do not establish.
- Distinguish basic, median-based, randomized, and Paranoid Quicksort.
- Cite relevant claims using [MIT Lecture 6, p. X] when available.
- Clearly state when the excerpts do not establish a claim.
- Explain concepts clearly for a university student.

Use exactly these three sections, with no numbering:

## ANSWER
Explain the concept clearly in 2–4 paragraphs. Include the algorithm's main steps and a small example when the excerpts support one.

## WHAT THE LECTURE STATES
Explain the relevant points established by the retrieved lecture excerpts. Include the important details and cite the source pages.

## WHAT THE EXCERPTS DO NOT ESTABLISH
Identify any relevant claims that cannot be confirmed from the supplied excerpts. Do not invent unsupported details.

LECTURE EXCERPTS:
{context}

STUDENT QUESTION:
{question}
"""

    client = genai.Client(
        api_key=api_key,
http_options=types.HttpOptions(timeout=60000),
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 500,
        },
    )

    generated_text = response.text

    if not generated_text:
        raise RuntimeError("Gemini returned an empty response.")

    return generated_text


# API for the Hugging Face frontend
app = FastAPI(title="AlgoLearn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks_loaded": len(CHUNKS)}


@app.post("/api/answer")
def api_answer(request: QuestionRequest):
    try:
        answer = answer_question(request.question)
        return {"answer": answer}
    except Exception as error:
        # Log the actual error in Render without exposing it to visitors.
        print(f"API generation error: {type(error).__name__}: {error}", flush=True)
        raise HTTPException(
            status_code=502,
            detail=f"Answer generation failed ({type(error).__name__}). Check Render logs."
        )


# Keep the existing Gradio interface available at /gradio
demo = gr.Interface(
    fn=lambda question: answer_question(question),
    inputs=gr.Textbox(
        label="Ask AlgoLearn",
        placeholder="Explain randomized Quicksort...",
        lines=3,
    ),
    outputs=gr.Markdown(label="Answer"),
    title="AlgoLearn — Your DSA Tutor",
    description=(
        "Ask questions about Quicksort. Answers use retrieved passages "
        "from MIT Lecture 6 and include source references."
    ),
    examples=[
        "How does randomized Quicksort work?",
        "What is the difference between randomized and basic Quicksort?",
        "What does the lecture establish about expected running time?",
    ],
)

app = gr.mount_gradio_app(app, demo, path="/gradio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
