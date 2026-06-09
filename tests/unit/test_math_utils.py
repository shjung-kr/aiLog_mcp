import math

import pytest

from app.utils.math import batch_cosine_similarity, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(cosine_similarity(v, v), 1.0, abs_tol=1e-6)

    def test_orthogonal_vectors(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-6)

    def test_opposite_vectors(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, abs_tol=1e-6)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_empty_input_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_known_value(self):
        # [1,1] vs [1,0] → cos(45°) = 1/√2 ≈ 0.7071
        result = cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert math.isclose(result, 1 / math.sqrt(2), abs_tol=1e-5)


class TestBatchCosineSimilarity:
    def test_single_identical(self):
        q = [1.0, 0.0]
        scores = batch_cosine_similarity(q, [[1.0, 0.0]])
        assert math.isclose(scores[0], 1.0, abs_tol=1e-6)

    def test_multiple_rows(self):
        q = [1.0, 0.0]
        matrix = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = batch_cosine_similarity(q, matrix)
        assert math.isclose(scores[0], 1.0, abs_tol=1e-6)
        assert math.isclose(scores[1], 0.0, abs_tol=1e-6)
        assert math.isclose(scores[2], -1.0, abs_tol=1e-6)

    def test_empty_matrix_returns_empty(self):
        assert batch_cosine_similarity([1.0, 0.0], []) == []

    def test_empty_query_returns_zeros(self):
        scores = batch_cosine_similarity([], [[1.0, 0.0]])
        assert scores == [0.0]

    def test_dimension_mismatch_scores_zero(self):
        scores = batch_cosine_similarity([1.0, 0.0], [[1.0, 0.0, 0.0]])
        assert scores[0] == 0.0
