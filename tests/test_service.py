import pytest

from rag.generation.context import ContextBuilder
from rag.generation.prompt import PromptBuilder
from rag.models import Document, RetrievalResult
from rag.service import RAGService


class FakeRetriever:
    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                document=Document(
                    content="Overfitting occurs when a model ",
                    metadata={
                        "source": "ml.pdf",
                        "page": 7
                    }
                ),
                score=0.9
            )
        ]

class FaakeLLM:
    def generate(
            self,
            prompt: str
    ) -> str:
        return (
            "Overfitting occurs when a model learns "
            "training data too closely [Source 1]."
        )


def test_rag_service_returns_answer_and_sources():
    service = RAGService(
        retriever=FakeRetriever(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FaakeLLM(),
        top_k=5
    )

    response = service.ask(
        "What is overfitting?"
    )

    assert "Overfitting" in response.answer
    assert "[Source 1]" in response.answer

    assert len(response.sources) == 1
    assert (
        response.sources[0].document.metadata["page"] == 7
    )


def test_tag_service_rejects_empty_quesitons():
    service = RAGService(
        retriever=FakeRetriever(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FaakeLLM(),
        top_k=5
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty"
    ):
        service.ask(" ")


def test_rag_service_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be grater than 0"
    ):
        RAGService(
            retriever=FakeRetriever(),
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm=FaakeLLM(),
            top_k=0
        )