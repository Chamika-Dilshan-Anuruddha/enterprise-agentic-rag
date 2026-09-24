import pytest

from rag.models import Document, RetrievalResult
from rag.retrieval.fusion import reciprocal_rank_fusion


def make_result(
        document: Document,
        score: float
) -> RetrievalResult:
    return RetrievalResult(
        document=document,
        score=score
    )


def test_empty_ranked_results():
    results = reciprocal_rank_fusion([])

    assert results == []

def test_invalid_k():
    with pytest.raises(
        ValueError,
        match="k must be grater than 0"
    ):
        reciprocal_rank_fusion(
            ranked_results=[],
            k=0
        )

def test_single_ranking_preserves_order():
    document_a = Document(content="A")
    document_b = Document(content="B")

    ranked_results = [
        [
            make_result(document_a, 0.9),
            make_result(document_b, 0.8),

        ]
    ]

    results = reciprocal_rank_fusion(
        ranked_results
    )

    assert results[0].document == document_a
    assert results[1].document == document_b


def test_document_receives_score_from_multiple_rankings():
    document_a = Document(content="A")
    document_b = Document(content="B")

    dense_results = [
        make_result(document_a, 0.95),
        make_result(document_b, 0.80),
    ]

    bm25_results = [
        make_result(document_b, 8.5),
        make_result(document_a, 6.2),
    ]

    results = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results
        ],
        k=60
    )

    expected_a = (
        1 /61
        + 1 / 62
    )

    assert results[0].score == pytest.approx(expected_a)


def test_document_ranked_hight_by_both_retrievers_win():
    document_a = Document(content="A")
    document_b = Document(content="B")
    document_c = Document(content="C")

    dense_results = [
        make_result(document_a, 0.95),
        make_result(document_b, 0.80),
        make_result(document_c, 0.80),
    ]

    bm25_results = [
        make_result(document_a, 6.2),
        make_result(document_b, 8.5),
        make_result(document_c, 5.0),
    ]

    results = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results
        ],
        k=60
    )   

    assert results[0].document == document_a   


def test_rff_uses_rank_not_raw_score():
    document_a = Document(content="A")
    document_b = Document(content="B")

    dense_results = [
        make_result(document_a, 0.95),
        make_result(document_b, 0.80),
    ]

    bm25_results = [
        make_result(document_a, 1.0),
        make_result(document_b, 1000.0),
    ]

    results = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results
        ],
        k=60
    )

    assert results[0].document == document_a

