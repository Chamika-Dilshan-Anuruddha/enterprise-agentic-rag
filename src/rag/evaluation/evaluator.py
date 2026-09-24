from dataclasses import dataclass

from rag.evaluation.retrieval_metrics import find_first_relevant_rank, hit_at_k, reciprocal_rank
from rag.models import QueryEvaluationResult, RetrievalEvaluationCase
from rag.retrieval.base import Retriever


@dataclass(slots=True)
class RetrievalMetricSummary:
    total_queries: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float

@dataclass(slots=True)
class RetrievalEvaluationReport:
    overall: RetrievalMetricSummary
    by_category: dict[str, RetrievalMetricSummary]
    query_results: list[QueryEvaluationResult]


def calculate_summary(
        results: list[QueryEvaluationResult]
) -> RetrievalMetricSummary:
    if not results:
        raise ValueError("Results cannot be empty")

    total = len(results)

    return RetrievalMetricSummary(
        total_queries=total,
        hit_rate_at_1=sum(result.hit_at_1 for result in results) / total,
        hit_rate_at_3=sum(result.hit_at_3 for result in results) / total,
        hit_rate_at_5=sum(result.hit_at_5 for result in results) / total,
        mrr=sum(result.reciprocal_rank for result in results) / total,
    )


class RetrievalEvaluator:
    """Evaludate retrieval quality agains groud-truth cases."""

    def __init__(
            self,
            retriever: Retriever
    ):
        self.retriever = retriever

    def evaluate(
            self,
            cases: list[RetrievalEvaluationCase]
    ) -> RetrievalEvaluationReport:
        if not cases:
            raise ValueError("Evaluation cases cannot be empty")

        query_results = []

        for case in cases:
            results = self.retriever.retrieve(
                query=case.query,
                top_k=5
            )

            query_hit_at_1 = hit_at_k(
                results,
                relevant_text=case.relevant_text,
                k=1
            )
            query_hit_at_3 = hit_at_k(
                results,
                relevant_text=case.relevant_text,
                k=3
            )
            query_hit_at_5 = hit_at_k(
                results,
                relevant_text=case.relevant_text,
                k=5
            )
            query_rr = reciprocal_rank(
                results,
                relevant_text=case.relevant_text
            )

            rank = find_first_relevant_rank(
                results,
                relevant_text=case.relevant_text
            )

            query_results.append(
                QueryEvaluationResult(
                    id=case.id,
                    category=case.category,
                    query=case.query,
                    relevant_text=case.relevant_text,
                    first_relevant_rank=rank,
                    hit_at_1=query_hit_at_1,
                    hit_at_3=query_hit_at_3,
                    hit_at_5=query_hit_at_5,
                    reciprocal_rank=query_rr
                )
            )

        categories = {
            result.category for result in query_results
        }

        by_category = {}

        for category in categories:
            category_results = [
                result 
                for result in query_results
                if result.category == category
            ]

            by_category[category] = calculate_summary(
                category_results
            )

        return RetrievalEvaluationReport(
            overall=calculate_summary(query_results),
            by_category=by_category,
            query_results=query_results
        )
