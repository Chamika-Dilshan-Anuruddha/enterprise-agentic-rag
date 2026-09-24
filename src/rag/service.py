from rag.generation.context import ContextBuilder
from rag.generation.prompt import PromptBuilder
from rag.llm.base import LLMProvider
from rag.models import RAGResponse
from rag.retrieval.base import Retriever


class RAGService:
    """Orchestrate retrieval and grounded answer generation."""

    def __init__(
            self,
            retriever: Retriever,
            context_builder: ContextBuilder,
            prompt_builder: PromptBuilder,
            llm: LLMProvider,
            top_k: int = 5
    ):
        if top_k <= 0:
            raise ValueError("top_k must be grater than 0")

        self.retriever = retriever
        self.context_builder = context_builder
        self.promt_builder = prompt_builder
        self.llm = llm
        self.top_k = top_k


    def ask(
            self,
            question: str
    ) -> RAGResponse:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        results = self.retriever.retrieve(
            query=question,
            top_k=self.top_k
        )

        context = self.context_builder.build(
            results
        )

        prompt = self.promt_builder.build(
            question=question,
            context=context
        )

        answer = self.llm.generate(
            prompt
        )

        return RAGResponse(
            answer=answer,
            sources=results
        )