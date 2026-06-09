import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on length mismatch or zero norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def batch_cosine_similarity(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Cosine similarity between a query vector and each row of a matrix.

    Returns a list of scores in the same order as matrix rows.
    Rows with dimension mismatch silently score 0.0.
    """
    if not query or not matrix:
        return [0.0] * len(matrix)

    dim = len(query)
    vq = np.array(query, dtype=np.float32)
    norm_q = np.linalg.norm(vq)
    if norm_q == 0:
        return [0.0] * len(matrix)

    scores: list[float] = []
    for row in matrix:
        if not row or len(row) != dim:
            scores.append(0.0)
            continue
        vr = np.array(row, dtype=np.float32)
        norm_r = np.linalg.norm(vr)
        if norm_r == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(vq, vr) / (norm_q * norm_r)))
    return scores
