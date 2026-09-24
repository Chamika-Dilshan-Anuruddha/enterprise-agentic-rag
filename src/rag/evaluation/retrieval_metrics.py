from rag.models import RetrievalResult


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())

def find_first_relevant_rank(
        results: list[RetrievalResult],
        relevant_text: str
) -> int | None:
    """Return the 1-based rank of the first relavent result."""

    expected = relevant_text.lower()

    for rank, result in enumerate(results, start=1):
        content = result.document.content.lower()

        if expected in normalize_text(content):
            return rank

    return None


def hit_at_k(
        results: list[RetrievalResult],
        relevant_text: str,
        k: int
) -> float:
    """Return 1.0 if a relevant result occurs within top-k"""

    if k <= 0:
        raise ValueError("k must be grater than 0")

    rank = find_first_relevant_rank(
        results[:k],
        relevant_text
    )

    return 1.0 if rank is not None else 0.0


def reciprocal_rank(
        results: list[RetrievalResult],
        relevant_text: str,
) -> float:
    """Return reciporcal rank of the first relevant result."""

    rank = find_first_relevant_rank(
        results,
        relevant_text
    )

    if rank is None:
        return 0.0

    return 1.0 / rank