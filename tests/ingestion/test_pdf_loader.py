from pathlib import Path

import pytest

from rag.ingestion.pdf import PDFLoader


def test_missing_pdf_raises_file_not_found():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("does-not-exist.pdf")

def test_non_pdf_file_raises_value_error(tmp_path: Path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Hello RAG", encoding="utf-8")

    loader = PDFLoader()
    with pytest.raises(ValueError):
        loader.load(text_file)