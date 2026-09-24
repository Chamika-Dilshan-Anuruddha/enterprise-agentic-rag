import pytest

from rag.models import Document
from rag.retrieval.bm25 import BM25Retriever, tokenize


def test_tokenize():
    tokens = tokenize("Machine Learning!")

    assert tokens == ["machine", "learning"]

def test_average_document_length():
    documents = [
        Document(content="machine learning"),
        Document(content="python is very usefull"),
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    assert retriever.average_document_length == pytest.approx(3.0)


def test_document_frequency_countes_document_not_occurrences():
    documents = [
        Document(content="machine learning"),
        Document(content="python python python"),
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    assert retriever.document_frequencies["python"] == 1


def test_exact_lexical_match_ranks_higher():
    python_document = Document(
        content="Python is a programming language"
    )

    unrelated_document = Document(
        content="The weather is sunny today"
    )

    retriever = BM25Retriever()
    retriever.index([
        python_document,
        unrelated_document
    ])

    results =retriever.retrieve(
        query="python",
        top_k=2
    )

    assert results[0].document == python_document
    assert results[0].score > results[1].score


def test_rare_term_has_higher_idf_than_common_term():
    documents = [
        Document(content="learning python"),
        Document(content="learning machine"),
        Document(content="learning statistics"),
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    common_idf = retriever._idf("learning")
    rare_idf = retriever._idf("pthon")

    assert rare_idf > common_idf


def test_invalid_top_k():
    documents = [
        Document(content="Python programming")
    ]

    retriever = BM25Retriever()
    retriever.index(documents)

    with pytest.raises(ValueError, match="top_k mush be grater thatn 0"):
        retriever.retrieve(
            query="python",
            top_k=0
        )

def test_retrieve_before_index():
    retriever = BM25Retriever()

    with pytest.raises(RuntimeError, match="No documents have been indexed"):
        retriever.retrieve(
            query="python"
        )


def test_invalid_k1():
    with pytest.raises(ValueError, match="k1 mest be grater thatn 0"):
        BM25Retriever(k1=0)


def test_invalid_b():
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Retriever(b=-0.5)  