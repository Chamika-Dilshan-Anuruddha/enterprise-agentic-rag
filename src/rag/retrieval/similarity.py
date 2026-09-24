import numpy as np


def consine_similarity(
        vector_a: list[float],
        vector_b: list[float]
) -> float:

    """Calculate consine similarityy between two vectors."""
    a = np.asanyarray(vector_a)
    b = np.asanyarray(vector_b)

    if a.shape != b.shape:
        raise ValueError("Vectors must have the same dimenstions")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector")

    similarity = np.dot(a, b) / (norm_a * norm_b)

    return float(similarity)

