from rag.models import RetrievalResult


def reciprocal_rank_fusion(
        ranked_results: list[list[RetrievalResult]],
        k: int = 60
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists using Resciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("k must be grater than 0")

    if not ranked_results:
        return []

    fused_scores: dict[int, float] = {}
    documents = {}

    for results in ranked_results:
        for rank, result in enumerate(results, start=1):
            document_id = id(result.document)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

            documents[document_id] = result.document

    fused_results = [
        RetrievalResult(
            document=documents[document_id],
            score=score
        )
        for document_id, score in fused_scores.items()
    ]

    fused_results.sort(
        key=lambda result: result.score,
        reverse=True
    )

    return fused_results
