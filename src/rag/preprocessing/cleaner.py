import re

from rag.models import Document


class TextCleaner:
    """Normalize extracted document text while preserving useful structure."""

    def clean(self, document: Document) -> Document:
        text = document.content

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return Document(
            content=text,
            metadata=document.metadata.copy()
        )