import pytest

from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.models import Document


def test_short_document_creates_one_chunk():
    document = Document(
        content="Retrieval augmented generation uses external knowledge.",
        metadata={"page": 1},
    )

    chunker = LangChainRecursiveChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content


def test_long_document_creates_multiple_chunks():
    document = Document(
        content="Retrieval augmented generation improves LLM< responses." * 20,
        metadata={"page": 1},
    )

    chunker = LangChainRecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
        )
    
    chunks = chunker.split(document)

    assert len(chunks) > 1


def test_preserves_metadata():
    document = Document(
        content="RAG evaluation is important. " * 20,
        metadata={
            "source": "rag.pdf",
            "page": 4
        },  
    )

    chunker = LangChainRecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
        )
    
    chunks = chunker.split(document)

    assert chunks[0].metadata["source"] == "rag.pdf"
    assert chunks[0].metadata["page"] == 4
    assert chunks[0].metadata["chunking_strategy"] == "langchain_recursive"


def test_invalid_overlap_raises_value_error():
    with pytest.raises(ValueError):
        LangChainRecursiveChunker(
            chunk_size=100,
            chunk_overlap=100
        )


def test_empty_document_returns_no_chuks():
    document = Document(
        content="",
        metadata={"page": 1}
    )
    chunker = LangChainRecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
        )
    
    chunks = chunker.split(document)

    assert chunks == []