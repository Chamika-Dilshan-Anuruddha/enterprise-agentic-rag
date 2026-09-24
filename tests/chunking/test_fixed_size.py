import pytest

from rag.chunking.fixed_size import FixedSizeChunker
from rag.models import Document


def test_short_document_creates_one_chunk():
    document = Document(
        content="Hello RAG",
        metadata={"page": 1}
    )

    chunker = FixedSizeChunker(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = chunker.split(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello RAG"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunker_creates_overlap():
    document = Document(
        content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        metadata={"page": 1}
    )

    chunker = FixedSizeChunker(
        chunk_size=10,
        chunk_overlap=2
    )

    chunks = chunker.split(document)

    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[1].content == "IJKLMNOPQR"

def test_chunker_preserves_metadata():
    document = Document(
        content="A" * 100,
        metadata={
            "source": "policy.pdf",
            "page": 12
        }
    )

    chunker = FixedSizeChunker(
        chunk_size=30,
        chunk_overlap=5
    )

    chunks = chunker.split(document)

    assert chunks[0].metadata["source"] == "policy.pdf"
    assert chunks[0].metadata["page"] == 12


def test_overlap_cannot_equal_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(
            chunk_size=100,
            chunk_overlap=100
        )
    