from rag.models import RetrievalResult
from rag.retrieval.base import Retriever
from rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Combine multiple retrievers using Reciprocal Rank Fusion."""

    def __init__(
            self,
            retrievers: list[Retriever],
            rrf_k: int = 60
    ):
        if not retrievers:
            raise ValueError("At least one retriever is required")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be grater thatn 0")

        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            raise ValueError("top_k must be grater than 0")

        ranked_results = [
            retriever.retrieve(
                query=query,
                top_k=top_k
            )
            for retriever in self.retrievers
        ]

        fesued_results = reciprocal_rank_fusion(
            ranked_results=ranked_results,
            k=self.rrf_k
        )

        return fesued_results[:top_k]