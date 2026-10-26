from pathlib import Path
import json
import re
from pypdf import PdfReader

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = PROJECT_DIR / "data" / "documents"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

PDF_NAME = "mit_lecture_06_quicksort.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_pages(pdf_path):
    """Extract text from a PDF while preserving page numbers."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def split_into_chunks(text):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def main():
    pdf_path = DOCUMENT_DIR / PDF_NAME

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Check the filename and documents folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(pdf_path)
    records = []

    for page_data in pages:
        chunks = split_into_chunks(page_data["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            records.append({
                "chunk_id": (
                    f"lecture06_page{page_data['page']}"
                    f"_chunk{chunk_number}"
                ),
                "source": PDF_NAME,
                "page": page_data["page"],
                "text": chunk
            })

    output_path = OUTPUT_DIR / "chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"PDF processed: {PDF_NAME}")
    print(f"Pages with extractable text: {len(pages)}")
    print(f"Total chunks created: {len(records)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()