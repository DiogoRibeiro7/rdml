"""Tests for private factorized positive-definite update primitives."""

from __future__ import annotations

import numpy as np
import pytest

from rdml._factorized import (
    FactorizationIllConditionedError,
    cholesky_rank_one,
    factor_relative_diagonal,
    forward_substitution,
    inverse_quadratic_from_factor,
)


def test_forward_substitution_matches_numpy_solve() -> None:
    lower = np.array([[2.0, 0.0], [0.5, 1.5]], dtype=np.float64)
    rhs = np.array([1.0, 2.0], dtype=np.float64)

    actual = forward_substitution(lower, rhs)

    np.testing.assert_allclose(actual, np.linalg.solve(lower, rhs), atol=1e-12)


def test_inverse_quadratic_matches_direct_matrix_solve() -> None:
    lower = np.array([[2.0, 0.0], [0.4, 1.2]], dtype=np.float64)
    difference = np.array([1.5, -0.3], dtype=np.float64)
    metric = lower @ lower.T

    actual = inverse_quadratic_from_factor(lower, difference)
    expected = float(difference @ np.linalg.solve(metric, difference))

    assert actual == pytest.approx(expected, rel=1e-12)


def test_rank_one_update_matches_direct_matrix_update() -> None:
    lower = np.array([[1.5, 0.0], [0.2, 1.1]], dtype=np.float64)
    vector = np.array([0.4, -0.3], dtype=np.float64)
    metric = lower @ lower.T

    updated = cholesky_rank_one(lower, vector, sign=1)

    np.testing.assert_allclose(
        updated @ updated.T,
        metric + np.outer(vector, vector),
        atol=1e-12,
    )


def test_rank_one_downdate_matches_direct_matrix_downdate() -> None:
    lower = np.array([[2.0, 0.0], [0.2, 1.5]], dtype=np.float64)
    vector = np.array([0.4, 0.2], dtype=np.float64)
    metric = lower @ lower.T

    updated = cholesky_rank_one(lower, vector, sign=-1)
    expected = metric - np.outer(vector, vector)

    np.testing.assert_allclose(updated @ updated.T, expected, atol=1e-12)
    assert np.linalg.eigvalsh(expected).min() > 0.0


def test_downdate_rejects_spd_boundary() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="SPD cone"):
        cholesky_rank_one(np.eye(2), [1.0, 0.0], sign=-1)


def test_conditioning_guard_rejects_tiny_existing_diagonal() -> None:
    lower = np.diag([1.0, 1e-14])

    assert factor_relative_diagonal(lower) == pytest.approx(1e-14)
    with pytest.raises(FactorizationIllConditionedError, match="below guard"):
        forward_substitution(
            lower,
            [1.0, 1.0],
            minimum_relative_diagonal=1e-12,
        )


def test_conditioning_guard_rejects_near_boundary_downdate() -> None:
    vector = np.array([np.sqrt(1.0 - 1e-14), 0.0])

    with pytest.raises(FactorizationIllConditionedError, match="below guard"):
        cholesky_rank_one(
            np.eye(2),
            vector,
            sign=-1,
            minimum_relative_diagonal=1e-6,
        )


def test_factorized_helpers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        factor_relative_diagonal(np.ones((2, 3)))
    with pytest.raises(ValueError, match="match the factor size"):
        forward_substitution(np.eye(2), [1.0])
    with pytest.raises(ValueError, match="minimum_relative_diagonal"):
        forward_substitution(
            np.eye(2),
            [1.0, 1.0],
            minimum_relative_diagonal=1.0,
        )
    with pytest.raises(ValueError, match="sign"):
        cholesky_rank_one(np.eye(2), [1.0, 0.0], sign=0)  # type: ignore[arg-type]
    with pytest.raises(np.linalg.LinAlgError, match="positive diagonal"):
        forward_substitution(np.diag([1.0, 0.0]), [1.0, 1.0])
