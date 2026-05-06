import pdfplumber
from pathlib import Path


def load_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Extract raw text and metadata from each page of a PDF.
    Returns a list of dicts: {page_num, text, tables}
    """
    pages = []
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append({
                "page_num": i + 1,
                "text": text,
                "tables": tables,
                "source": path.name,
            })

    return pages


def combine_pages(pages: list[dict]) -> str:
    """Flatten all page text into a single string."""
    return "\n\n".join(p["text"] for p in pages if p["text"].strip())