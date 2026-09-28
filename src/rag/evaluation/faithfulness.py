
from rag.llm.base import LLMProvider
from rag.models import FaithfulnessEvaluation


class FaithfullnessEvaluator:
    """Evaluation wether an answer is supported by its context."""

    def __init__(
            self,
            llm: LLMProvider
    ):
        self.llm = llm

    def evaluate(
            self,
            question: str,
            answer: str,
            context: str
    ) -> FaithfulnessEvaluation:
        prompt = self._build_prompt(
            question=question,
            answer=answer,
            context=context
        )

        return self.llm.generate_structured(
            prompt=prompt,
            response_model=FaithfulnessEvaluation
        )

    @staticmethod
    def _build_prompt(
        question: str,
        answer: str,
        context: str
    ) -> str:
        return (
            "Evaluation weather the ANSWER is supported by the CONTEXT.\n\n"
            "Do not evaluate using outside knowledge.\n"
            "Judge only whether claims in the answer are supported "
            "by the supplied context.\n\n"
            "Returen valid JSON only using this format:\n"
            '{"score": 0.0, "reasoning": "explanation"}\n\n'
            "The score must be betweeen 0.0 and 1.0.\n"
            "1.0 means fully supported by the context.\n"
            "0.0 means unsupported by the context.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}"
        )