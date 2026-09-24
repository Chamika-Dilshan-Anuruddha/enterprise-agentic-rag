import pytest

from rag.evaluation.evaluator import RetrievalEvaluator
from rag.models import Document, RetrievalEvaluationCase, RetrievalResult


class FakeRetriever:
    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:
        if query == "query one":
            return [
                RetrievalResult(
                    document=Document(content="correct answer one"),
                    score=0.9
                ),
                RetrievalResult(
                    document=Document(content="irrelevent"),
                    score=0.8
                ),
            ]
        
        return [
            RetrievalResult(
                document=Document(content="irrelevent"),
                score=0.9
            ),
            RetrievalResult(
                document=Document(content="correct answer two"),
                score=0.8
            ),
        ]


def test_evaluator_claculates_metrics():
    cases = [
        RetrievalEvaluationCase(
            id="semantic_001",
            category="semantic",
            query="query one",
            relevant_text="correct answer one"
        ),
        RetrievalEvaluationCase(
            id="direct_001",
            category="direct_lookup",
            query="query two",
            relevant_text="correct answer two"
        )
    ]

    evaluator = RetrievalEvaluator(retriever=FakeRetriever())

    report = evaluator.evaluate(cases)

    assert report.overall.total_queries == 2

    assert report.overall.hit_rate_at_1 == pytest.approx(0.5)
    assert report.overall.hit_rate_at_3 == pytest.approx(1.0)
    assert report.overall.hit_rate_at_5 == pytest.approx(1.0)
    assert report.overall.mrr == pytest.approx(0.75)


def test_evaluator_rejects_empty_cases():
    evaluator = RetrievalEvaluator(retriever=FakeRetriever())

    with pytest.raises(ValueError, match="Evaluation cases cannot be empty"):
        evaluator.evaluate([])

