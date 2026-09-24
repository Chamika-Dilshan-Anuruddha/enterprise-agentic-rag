from typing import Protocol

from rag.models import Document


class EmbeddingProvider(Protocol):
    """Interface reqired by components that generate embeddings."""

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_documents(
            self,
            documents: list[Document]
    ) -> list[list[float]]:
        ...