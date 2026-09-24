from rag.models import Document
from rag.preprocessing.cleaner import TextCleaner


def test_cleaner_normalizes_whitespace():
    document = Document(
        content="Hello      World.\n\n\n\nThis    is  RAG.",
        metadata={"page": 1}
    )

    cleaner = TextCleaner()

    result = cleaner.clean(document)

    assert result.content == "Hello World.\n\nThis is RAG."


def test_cleaner_preserves_metadata():
    document = Document(
        content="Sone text",
        metadata={
            "source": "sample.pdf",
            "page": 7
        }
    )

    cleaner = TextCleaner()

    result = cleaner.clean(document)

    assert result.metadata["source"] == "sample.pdf"
    assert result.metadata["page"] == 7