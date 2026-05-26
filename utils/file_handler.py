"""File handling utilities for TXT, PDF, and DOCX."""

import io
from typing import Optional

import docx
import PyPDF2


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a TXT file."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("No text could be extracted from the DOCX file.")
    return "\n\n".join(paragraphs)


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from an uploaded file based on extension."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if file_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if file_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    raise ValueError("Unsupported file type. Please upload TXT, PDF, or DOCX.")


def get_supported_file_types() -> list[str]:
    """Return list of supported file extensions."""
    return ["txt", "pdf", "docx"]
