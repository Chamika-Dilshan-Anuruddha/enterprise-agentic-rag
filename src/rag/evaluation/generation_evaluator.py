from dataclasses import dataclass

from rag.evaluation.faithfulness import FaithfullnessEvaluator
from rag.service import RAGService


@dataclass(slots=True)
class GenerationEvaluationResult:
    query: str
    answer: str
    faithfulness: float
    reasoning: str


@dataclass(slots=True)
class GenerationEvaluationSummary:
    results: list[GenerationEvaluationResult]
    mean_faithfulness: float
    min_faithfulness: float
    pass_rate: float


class GenerationEvaluator:
    def __init__(
            self,
            faithfulness_evaluator: FaithfullnessEvaluator,
            threshold: float = 0.8
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        self.faithfulness_evaluator = faithfulness_evaluator
        self.threshold = threshold


    def evaluate(
            self,
            queries: list[str],
            service: RAGService,
    ) -> GenerationEvaluationSummary:
        if not queries:
            raise ValueError("queries cannot be empty")

        results = []
        for query in queries:
            response = service.ask(query)

            evaluation = self.faithfulness_evaluator.evaluate(
                question=query,
                answer=response.answer,
                context=response.context
            )

            results.append(
                GenerationEvaluationResult(
                    query=query,
                    answer=response.answer,
                    faithfulness=evaluation.score,
                    reasoning=evaluation.reasoning
                )
            )

        scores = [
            result.faithfulness 
            for result in results
        ]

        mean_faithfulness = sum(scores) / len(scores)
        min_faithfulness = min(scores)

        passed = sum(
            score >= self.threshold
            for score in scores
        )

        pass_rate = passed / len(scores)

        return GenerationEvaluationSummary(
            results=results,
            mean_faithfulness=mean_faithfulness,
            min_faithfulness=min_faithfulness,
            pass_rate=pass_rate
        )



        