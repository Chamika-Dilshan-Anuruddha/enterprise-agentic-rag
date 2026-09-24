from typing import Protocol

from rag.models import RetrievalResult


class Retriever(Protocol):
    """Interface required by retrieval evaluation components."""

    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:
        ...