import pytest

from rag.models import Document, RetrievalResult
from rag.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    def __init__(
            self,
            results: list[RetrievalResult]
    ):
        self.results = results

    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:
        return self.results[:top_k]

def make_results(
        document: Document,
        score: float
) -> RetrievalResult:
    return RetrievalResult(
        document=document,
        score=score
    )


def test_combines_multiple_retrievers():
    document_a = Document(content="A")
    document_b = Document(content="B")
    document_c = Document(content="C")

    retriever_1 = FakeRetriever(
        [
            make_results(document_a, 0.9),
            make_results(document_b, 0.8),
            make_results(document_c, 0.7),
        ]
    )

    retriever_2 = FakeRetriever(
        [        
            make_results(document_a, 10.0),
            make_results(document_c, 8.9),
            make_results(document_b, 5.9),
        ]
    )

    hybrid = HybridRetriever(
        [
            retriever_1,
            retriever_2
        ]
    )

    results = hybrid.retrieve(
        query="test",
        top_k=3
    )

    assert results[0].document == document_a



def test_returns_only_top_k_results():
    document_a = Document(content="A")
    document_b = Document(content="B")
    document_c = Document(content="C")

    retriever_1 = FakeRetriever(
       [     
            make_results(document_a, 0.9),
            make_results(document_b, 0.8),
            make_results(document_c, 0.7),
        ]
    )

    hybrid = HybridRetriever(
        [
            retriever_1,
        ]
    )

    results = hybrid.retrieve(
        query="test",
        top_k=2
    )

    assert len(results) == 2


def test_requires_at_least_one_retriever():
    with pytest.raises(
        ValueError,
        match="At least one retriever is required"
    ):
        HybridRetriever([])


def test_invalid_rrf_k():
    retriever = FakeRetriever([])

    with pytest.raises(
        ValueError,
        match="rrf_k must be grater thatn 0"
    ):
        HybridRetriever(
            [retriever],
            rrf_k=0
        )


def test_invalid_top_k():
    retriever = FakeRetriever([])

    hybrid = HybridRetriever([
        retriever
    ])

    with pytest.raises(
        ValueError,
        match="top_k must be grater than 0"
    ):
        hybrid.retrieve(
            query="test",
            top_k=0
        )
        