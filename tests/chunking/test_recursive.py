import pytest

from rag.chunking.recursive import RecursiveChunker
from rag.models import Document


def test_short_document_creates_one_chunk():
    document = Document(
        content="Retrieval augmented generation uses external knwoledge.",
        metadata={"page": 1}
    )

    chunker = RecursiveChunker(
        chunk_size=1000,
        chunk_overlap=20
    )

    chunks = chunker.split(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content


def test_preserves_matadata():
    document = Document(
        content="RAG evaluaiton is important. " * 20,
        metadata={
            "source": "rag.pdf",
            "page": 3
        }
    )

    chunker = RecursiveChunker(
        chunk_size=1000,
        chunk_overlap=20
    )

    chunks = chunker.split(document)

    assert chunks[0].metadata["source"] == "rag.pdf"
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["chunking_strategy"] == "recursive"


def test_overlap_cannot_queal_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100
        )
