from pathlib import Path

import pymupdf

from rag.models import Document


class PDFLoader:
    """Load text-based PDF files into page-level documents."""

    def load(self, file_path: str | Path) -> list[Document]: 
        path = Path(file_path)

        self._validate_path(path)

        documents: list[Document] = []

        with pymupdf.open(path) as pdf:

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                document = Document(
                    content=text,
                    metadata={
                        "source": path.name,
                        "source_path": str(path),
                        "page": page_index + 1,
                        "document_type": "pdf"
                    }
                )

                documents.append(document)

        return documents

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {path.suffix}")