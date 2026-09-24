import pytest

from rag.evaluation.retrieval_metrics import find_first_relevant_rank, hit_at_k, reciprocal_rank
from rag.models import Document, RetrievalResult


def make_result(content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        document=Document(content=content),
        score=score
    )

def test_find_first_relevant_rank():
    results = [
        make_result("irrelevant document", 0.9),
        make_result("contains the target information", 0.8),
        make_result("another document", 0.7),
    ]

    rank = find_first_relevant_rank(
        results,
        "target information"
    )

    assert rank == 2

def test_missing_relevant_result_returns_none():
    results = [
        make_result("irrelevant document", 0.9),
        make_result("another document", 0.7),
    ]

    rank = find_first_relevant_rank(
        results,
        "target information"
    )

    assert rank is None

def test_hit_at_k_returns_one_when_found():

    results = [
        make_result("irrelevant document", 0.9),
        make_result("contains the target information", 0.7),
    ]

    score = hit_at_k(
       results,
       relevant_text="target information",
       k=2
    )

    assert score == 1.0


def test_hit_at_k_returns_zero_when_outsie_k():

    results = [
        make_result("irrelevant document", 0.9),
        make_result("contains the target information", 0.7),
    ]

    score = hit_at_k(
       results,
       relevant_text="target information",
       k=1
    )

    assert score == 0.0


def test_reciprocal_rank():

    results = [
        make_result("irrelevant document", 0.9),
        make_result("contains the target information", 0.7),
    ]

    score = reciprocal_rank(
       results,
       relevant_text="target information",
    )

    assert score == pytest.approx(0.5)


def test_reciprocal_rank_returns_zero_when_missing():

    results = [
        make_result("irrelevant document", 0.9),
        make_result("another information", 0.7),
    ]

    score = reciprocal_rank(
       results,
       relevant_text="target information",
    )

    assert score == 0.0


def test_hit_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        hit_at_k(
            results=[],
            relevant_text="aything",
            k=0
        )

