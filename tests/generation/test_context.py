from rag.generation.context import ContextBuilder
from rag.models import Document, RetrievalResult


def test_bild_context_from_single_result():
    document = Document(
        content="Overfitting occurs when model try to remember the training data.",
        metadata={
            "source": "machine_learning.pdf",
            "page": 7
        }
    )

    result = RetrievalResult(
        document=document,
        score=0.7
    )

    builder = ContextBuilder()

    context = builder.build([result])

    assert "[Source 1]" in context
    assert "File: machine_learning.pdf" in context
    assert "Page: 7" in context
    assert "Overfitting occurs" in context

def test_build_context_from_multiple_results():
    result_1 = RetrievalResult(
        document=Document(
            content="first document",
            metadata={
                "source": "doc_1.pdf",
                "page": 1
            }
        ),
        score=0.5
    )

    result_2 = RetrievalResult(
        document=Document(
            content="second document",
            metadata={
                "source": "doc_2.pdf",
                "page": 3
            }
        ),
        score=0.7
    )

    builder = ContextBuilder()

    context = builder.build([
        result_1,
        result_2
    ])

    assert "[Source 1]" in context
    assert "[Source 2]" in context

    assert "doc_1.pdf" in context
    assert "doc_2.pdf" in context

    assert "Page: 1" in context
    assert "Page: 3" in context


def test_missing_metadata_uses_unknown():
    document = Document(
        content="Some information"
    )

    result = RetrievalResult(
        document=document,
        score=0.4
    )

    builder = ContextBuilder()

    context = builder.build([result])

    assert "File: Unknown" in context
    assert "Page: Unknown" in context


def test_empty_results_returns_empty_context():
    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""