from typing import TypeVar

from pydantic import BaseModel

from rag.evaluation.faithfulness import FaithfullnessEvaluator

T = TypeVar("T", bound=BaseModel)

class FakeJudgeLLM:
    def generate_structured(
            self,
            prompt: str,
            response_model: type[T]
    ) -> T:
        return response_model(
            score=0.9,
            reasoning="The answer is supported by the context."
        )

def test_faithfulness_evaluator():
    evaluator = FaithfullnessEvaluator(llm=FakeJudgeLLM())

    result = evaluator.evaluate(
        question="What is overfitting?",
        answer=(
            "Overfitting happens when a model fails to generalize."
        ),
        context=(
            "Overfitting occurs when a model learns "
            "training data too closely and performs "
            "poorly on unseen data."
        )
    )

    assert result.score == 0.9
    assert "supported" in result.reasoning


def test_faithfulness_prompt_contains_inputs():
    evaluator = FaithfullnessEvaluator(llm=FakeJudgeLLM())

    prompt = evaluator._build_prompt(
        question="What is overfitting?",
        answer="Generated answer",
        context="Retrieved context."
    )

    assert "What is overfitting?" in prompt
    assert "Generated answer" in prompt
    assert "Retrieved context." in prompt



def test_faithfulness_score_is_parses_as_float():
    evaluator = FaithfullnessEvaluator(llm=FakeJudgeLLM())

    result = evaluator.evaluate(
        question="Question",
        answer="Answer",
        context="Context"
    )

    assert isinstance(result.score, float)