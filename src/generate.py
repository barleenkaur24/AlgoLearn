from pathlib import Path
import sys
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from retriever import load_chunks, search, MODEL_NAME as EMBEDDING_MODEL

PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_FILE = PROJECT_DIR / "data" / "processed" / "embeddings.npy"
LLM_MODEL = "llama3.2"
# Load the embedding model once and reuse it for every question.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def generate_answer(question):
    chunks = load_chunks()

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run src/retriever.py first."
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embeddings and chunks do not match. "
            "Rebuild embeddings using src/retriever.py."
        )

    # Retrieve relevant MIT lecture passages
    results = search(question, chunks, embedding_model, embeddings)

    context_parts = []
    for result in results:
        context_parts.append(
            f"[Source: {result['source']}, page {result['page']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are AlgoLearn, a DSA tutor using a restricted MIT lecture knowledge base.

SOURCE AND ACCURACY RULES:
1. Answer using only the supplied lecture excerpts.
2. Do not use outside knowledge, external books, or external references.
3. Never invent mathematical derivations or complexity analyses.
4. Cite supporting claims using [MIT Lecture 6, p. X].
5. If the excerpts state a result without proving it, explicitly say that
   the proof is not included in the provided excerpts.
6. Preserve every mathematical expression and numerical bound exactly.
7. If evidence is insufficient, say so instead of guessing.

ALGORITHM DISTINCTIONS:
- Basic Quicksort: The lecture states that sorted or reverse-sorted
  inputs can cause Theta(n^2) time when an extreme element is chosen
  as the pivot repeatedly.
- Median-based pivoting: The lecture says median finding can guarantee
  balanced partitions. Do not attribute this guarantee to Randomized
  Quicksort.
- Randomized Quicksort: The lecture states that a random element is
  chosen as the pivot and that expected time is O(n log n) for all
  input arrays. The supplied excerpts do not prove this bound.
- Paranoid Quicksort: This is a separate variant. It repeats random
  pivot selection until both partition sizes are at most 3/4 of the
  current array size, then recurses.
- Never combine the properties or analyses of these variants.
- Never say the lecture proves a result unless its proof is present
  in the supplied excerpts.

Do not claim that the excerpts prove something they merely state.
"""

    user_prompt = f"""
LECTURE EXCERPTS:
{context}

STUDENT QUESTION:
{question}

Follow this exact response structure:

ANSWER:
Answer the question using only claims directly supported by the excerpts.

WHAT THE LECTURE STATES:
List the relevant statements explicitly present in the excerpts.

WHAT THE EXCERPTS DO NOT ESTABLISH:
Identify any requested explanation or proof that is absent.
Do not invent a derivation or claim the lecture proves a result
unless the supplied text explicitly contains that proof.

IMPORTANT:
- Do not claim random pivots always produce balanced partitions.
- Do not claim the Paranoid Quicksort proof is present in these excerpts.
- The 3/4 partition condition applies only to Paranoid Quicksort.
- Do not cite external books, even if the lecture mentions them.
- Cite claims using [MIT Lecture 6, p. X].
"""


    # Generate an answer using the local Ollama model
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 220,
        },
    )

    answer = response["message"]["content"]
    return answer, results


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python src/generate.py '
            '"How does randomized Quicksort work?"'
        )
        return

    question = " ".join(sys.argv[1:])

    print("Generating an answer from the MIT lecture...\n")

    try:
        answer, sources = generate_answer(question)

        print("=" * 60)
        print("ALGOLEARN ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("RETRIEVED MIT SOURCES")
        print("=" * 60)

        for source in sources:
            print(
                f"- {source['source']}, page {source['page']} "
                f"(similarity: {source['score']:.4f})"
            )

    except Exception as error:
        print(f"\nSomething went wrong: {error}")
        print(
            "\nCheck that Ollama is running and that the model "
            f"'{LLM_MODEL}' is installed."
        )


if __name__ == "__main__":
    main()