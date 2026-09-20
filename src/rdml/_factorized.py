"""Private numerical primitives for factorized positive-definite metric updates."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class FactorizationIllConditionedError(np.linalg.LinAlgError):
    """Raised when a Cholesky factor crosses a requested conditioning guard."""


def _as_square_factor(values: ArrayLike) -> FloatArray:
    factor = np.asarray(values, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1] or factor.shape[0] == 0:
        raise ValueError("lower must be a non-empty square matrix.")
    if not np.all(np.isfinite(factor)):
        raise ValueError("lower must contain only finite values.")
    return factor


def _as_vector(values: ArrayLike, *, size: int) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size != size:
        raise ValueError("vector must be one-dimensional and match the factor size.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values.")
    return vector


def _validate_guard(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
        raise ValueError("minimum_relative_diagonal must be finite and in [0, 1).")
    return threshold


def factor_relative_diagonal(lower: ArrayLike) -> float:
    """Return min(diag(L)) / max(diag(L)) for a positive Cholesky factor."""
    factor = _as_square_factor(lower)
    diagonal = np.diag(factor)
    if np.any(diagonal <= 0.0):
        return 0.0
    return float(np.min(diagonal) / np.max(diagonal))


def _require_conditioned(factor: FloatArray, *, threshold: float) -> None:
    ratio = factor_relative_diagonal(factor)
    if ratio == 0.0:
        raise np.linalg.LinAlgError("lower must have a strictly positive diagonal.")
    if threshold > 0.0 and ratio < threshold:
        raise FactorizationIllConditionedError(
            f"factor relative diagonal {ratio:.3e} is below guard {threshold:.3e}."
        )


def forward_substitution(
    lower: ArrayLike,
    rhs: ArrayLike,
    *,
    minimum_relative_diagonal: float = 0.0,
) -> FloatArray:
    """Solve a lower-triangular system with an optional conditioning guard."""
    factor = _as_square_factor(lower)
    vector = _as_vector(rhs, size=factor.shape[0])
    threshold = _validate_guard(minimum_relative_diagonal)
    _require_conditioned(factor, threshold=threshold)

    solution = np.empty_like(vector)
    for row in range(vector.size):
        residual = vector[row] - factor[row, :row] @ solution[:row]
        solution[row] = residual / factor[row, row]
    return solution


def inverse_quadratic_from_factor(
    lower: ArrayLike,
    difference: ArrayLike,
    *,
    minimum_relative_diagonal: float = 0.0,
) -> float:
    """Return v.T @ (L L.T)^-1 @ v from a Cholesky factor L."""
    solved = forward_substitution(
        lower,
        difference,
        minimum_relative_diagonal=minimum_relative_diagonal,
    )
    return float(solved @ solved)


def cholesky_rank_one(
    lower: ArrayLike,
    vector: ArrayLike,
    *,
    sign: int,
    minimum_relative_diagonal: float = 0.0,
) -> FloatArray:
    """Return a Cholesky factor for A + sign * x x.T.

    The lower factor must satisfy A = lower @ lower.T. A positive sign performs
    a rank-one update; a negative sign performs a downdate. The optional guard
    aborts before the returned factor becomes too close to numerical singularity.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be either -1 or 1.")

    factor = _as_square_factor(lower)
    work = _as_vector(vector, size=factor.shape[0]).copy()
    threshold = _validate_guard(minimum_relative_diagonal)
    _require_conditioned(factor, threshold=threshold)
    updated = factor.copy()

    for index in range(work.size):
        diagonal = updated[index, index]
        radicand = diagonal * diagonal + sign * work[index] * work[index]
        if not np.isfinite(radicand) or radicand <= 0.0:
            raise np.linalg.LinAlgError("rank-one downdate left the SPD cone.")

        replacement = float(np.sqrt(radicand))
        updated[index, index] = replacement
        _require_conditioned(updated, threshold=threshold)

        cosine = replacement / diagonal
        sine = work[index] / diagonal
        if index + 1 < work.size:
            column = (
                updated[index + 1 :, index] + sign * sine * work[index + 1 :]
            ) / cosine
            updated[index + 1 :, index] = column
            work[index + 1 :] = cosine * work[index + 1 :] - sine * column

    return updated
