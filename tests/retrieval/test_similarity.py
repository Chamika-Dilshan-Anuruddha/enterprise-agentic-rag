import pytest

from rag.retrieval.similarity import consine_similarity


def test_identical_vectors_have_similarity_one():
    similarity = consine_similarity(
        [1.0, 0.0],
        [1.0, 0.0]
    )

    assert similarity == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    similarity = consine_similarity(
        [1.0, 0.0],
        [0.0, 1.0]
    )

    assert similarity == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    similarity = consine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0]
    )

    assert similarity == pytest.approx(-1.0)   


def test_different_dimensions_raise_error():
    with pytest.raises(ValueError):
        consine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0]

        ) 

def test_zero_vector_raises_zero():
    with pytest.raises(ValueError):
        consine_similarity(
            [0.0, 0.0],
            [1.0, 2.0]
        )