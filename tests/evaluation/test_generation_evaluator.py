from dataclasses import dataclass

import pytest

from rag.evaluation.generation_evaluator import GenerationEvaluator


@dataclass
class FakeRAGResponse:
    answer: str
    context: str

@dataclass
class FakeFaithfulnessResult:
    score: float
    reasoning: str

class FakeRAGService:
    def ask(self, question: str) -> FakeRAGResponse:
        return FakeRAGResponse(
            answer=f"Answer for {question}",
            context=f"Context for {question}"
        )

class FakeFaithfulnessEvaluator:
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str
    ) -> FakeFaithfulnessResult:
        return FakeFaithfulnessResult(
            score=0.9,
            reasoning="The answer is supported."
        )


def test_generation_evaluator():
    evaluator = GenerationEvaluator(
        faithfulness_evaluator=FakeFaithfulnessEvaluator(),
        threshold=0.8  
    )

    summary = evaluator.evaluate(
        queries=[
            "What is overfitting?",
            "What is underfitting"
        ],
        service=FakeRAGService()
    )

    assert len(summary.results) == 2

    assert summary.mean_faithfulness == pytest.approx(0.9)
    assert summary.min_faithfulness == pytest.approx(0.9)
    assert summary.pass_rate == pytest.approx(1.0)


def test_generation_evaluator_stores_per_query_results():
    evaluator = GenerationEvaluator(
        faithfulness_evaluator=FakeFaithfulnessEvaluator(),
        threshold=0.8
    )

    summary = evaluator.evaluate(
        queries=[
            "What is overfitting?",
            "What is underfitting"
        ],
        service=FakeRAGService()
    ) 

    result = summary.results[0]

    assert result.query == "What is overfitting?"  
    assert "Answer for" in result.answer
    assert result.faithfulness == pytest.approx(0.9)
    assert result.reasoning == "The answer is supported."


def test_generation_evaluator_rejects_empty_queries():
    evaluator = GenerationEvaluator(
        faithfulness_evaluator=FakeFaithfulnessEvaluator(),
        threshold=0.8
    )

    with pytest.raises(
        ValueError,
        match="queries cannot be empty"
    ):
        evaluator.evaluate(
            queries=[],
            service=FakeRAGService()
        )  


def test_generation_evaluator_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1"
    ):
        GenerationEvaluator(
            faithfulness_evaluator=FakeFaithfulnessEvaluator(),
            threshold=-0.5
        )

