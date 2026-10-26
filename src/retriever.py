from pathlib import Path
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_FILE = PROJECT_DIR / "data" / "processed" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def load_chunks():
    """Load text chunks and their source metadata."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Chunks not found. Run src/ingest.py first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_embeddings(chunks, model):
    """Generate and save embeddings for all document chunks."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saved embeddings for {len(chunks)} chunks.")
    return embeddings


def search(query, chunks, model, embeddings):
    """Retrieve the most relevant chunks for a question."""
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    # Normalized vectors allow dot product to measure cosine similarity.
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        result = chunks[int(index)].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python src/retriever.py '
            '"What is Quicksort?"'
        )
        return

    query = " ".join(sys.argv[1:])

    chunks = load_chunks()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    if EMBEDDINGS_FILE.exists():
        embeddings = np.load(EMBEDDINGS_FILE)

        if len(embeddings) != len(chunks):
            print("Chunk count changed. Rebuilding embeddings.")
            embeddings = build_embeddings(chunks, model)
    else:
        embeddings = build_embeddings(chunks, model)

    results = search(query, chunks, model, embeddings)

    print(f"\nQuestion: {query}")
    print("\nMost relevant passages:\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])
        print()


if __name__ == "__main__":
    main()