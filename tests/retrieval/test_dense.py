import pytest

from rag.models import Document
from rag.retrieval.dense import DenseRetriever


class FakeEmbedder:
    """Deterministic embedder for unit testing."""

    def embed_text(self, text:str) -> list[float]:
        embeddings = {
            "machine learning": [1.0, 0.0],
            "database systems": [0.0, 1.0],
            "ML query": [0.9, 0.1],
        }

        return embeddings[text]

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        return [
            self.embed_text(document.content)
            for document in documents
        ]

def test_retriever_returns_most_similar_document():
    documents = [
        Document(
            content="machine learning",
            metadata={"id": "ml"}
        ),
        Document(
            content="database systems",
            metadata={"id": "db"}
        ),
    ]

    retriever = DenseRetriever(embedder=FakeEmbedder())

    retriever.index(documents)

    results = retriever.retrieve(
        query="ML query",
        top_k=1
    )

    assert len(results) == 1
    assert results[0].document.metadata["id"] == "ml"


def test_retriever_respects_top_k():
    documents = [
        Document(
            content="machine learning",
            metadata={"id": "ml"}
        ),
        Document(
            content="database systems",
            metadata={"id": "db"}
        ),
    ]

    retriever = DenseRetriever(embedder=FakeEmbedder())

    retriever.index(documents)

    results = retriever.retrieve(
        query="ML query",
        top_k=2
    )

    assert len(results) == 2


def test_retriever_before_indexing_raises_error():
    retriever = DenseRetriever(embedder=FakeEmbedder())

    with pytest.raises(RuntimeError):
        retriever.retrieve(
            query="ML query",
            top_k=1
        )
    
def test_invalid_top_k_raises_error():
    documents = [
        Document(
            content="machine learning",
            metadata={"id": "ml"}
        ),
        Document(
            content="database systems",
            metadata={"id": "db"}
        ),
    ]

    retriever = DenseRetriever(embedder=FakeEmbedder())

    retriever.index(documents)

    with pytest.raises(ValueError):
        retriever.retrieve(
            query="ML query",
            top_k=0
        )
