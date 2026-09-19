"""Benchmark a factorized strict-interior RDML update on Wine.

This benchmark keeps the public estimator unchanged. It implements the same
strict-interior cap as the direct-solve reference, but represents the metric as

    A = L L^T

and applies rank-one Cholesky updates and downdates in O(d^2). This removes the
need for explicit inverses and per-step eigendecompositions while preserving a
direct comparison against the matrix-reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_wine

from rdml import accuracy_score, knn_predict
from rdml._factorized import cholesky_rank_one, forward_substitution

FloatMatrix = NDArray[np.float64]
IntLabels = NDArray[np.int64]
Preprocessing = Literal["raw", "standardized"]

N_RUNS = 10
N_NEIGHBORS = 3
LEARNING_RATE = 0.1
MAX_ITER = 10_000
MARGIN = 1.0
RHO = 0.1


@dataclass(frozen=True)
class FactorizedResult:
    """Repeated error and factor-conditioning diagnostics."""

    error_mean: float
    error_std: float
    zero_step_fraction: float
    minimum_diagonal_mean: float


def _split_half(
    X: FloatMatrix,
    y: IntLabels,
    *,
    seed: int,
) -> tuple[FloatMatrix, IntLabels, FloatMatrix, IntLabels]:
    """Return the fixed 50/50 split used by the reproduction suite."""
    order = np.random.default_rng(seed).permutation(X.shape[0])
    midpoint = X.shape[0] // 2
    train_idx = order[:midpoint]
    test_idx = order[midpoint:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _preprocess(
    X_train: FloatMatrix,
    X_test: FloatMatrix,
    *,
    mode: Preprocessing,
) -> tuple[FloatMatrix, FloatMatrix]:
    """Apply no scaling or train-only standardization."""
    if mode == "raw":
        return X_train, X_test

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    return (X_train - mean) / safe_scale, (X_test - mean) / safe_scale


def _fit_factorized(
    X: FloatMatrix,
    y: IntLabels,
    *,
    seed: int,
) -> tuple[FloatMatrix, int, int, float]:
    """Fit the strict-interior update entirely through a Cholesky factor."""
    n_samples, n_features = X.shape
    factor = np.eye(n_features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    similar_violations = 0
    zero_steps = 0
    minimum_diagonal = float("inf")

    for _ in range(MAX_ITER):
        first, second = rng.choice(n_samples, size=2, replace=False)
        difference = X[first] - X[second]
        pair_label = 1.0 if y[first] == y[second] else -1.0

        transformed_difference = difference @ factor
        distance = float(transformed_difference @ transformed_difference)
        if pair_label * (MARGIN - distance) > 0.0:
            continue

        if pair_label == -1.0:
            step = LEARNING_RATE
            factor = cholesky_rank_one(
                factor,
                np.sqrt(step) * difference,
                sign=1,
            )
        else:
            similar_violations += 1
            solved = forward_substitution(factor, difference)
            inverse_quadratic = float(solved @ solved)
            if not np.isfinite(inverse_quadratic) or inverse_quadratic <= 0.0:
                step = 0.0
                zero_steps += 1
            else:
                step = min(LEARNING_RATE, RHO / inverse_quadratic)

            if step > 0.0:
                factor = cholesky_rank_one(
                    factor,
                    np.sqrt(step) * difference,
                    sign=-1,
                )

        minimum_diagonal = min(
            minimum_diagonal,
            float(np.min(np.diag(factor))),
        )

    return factor, zero_steps, similar_violations, minimum_diagonal


def _evaluate_mode(
    X: FloatMatrix,
    y: IntLabels,
    *,
    preprocessing: Preprocessing,
) -> FactorizedResult:
    """Evaluate the factorized path over the ten fixed Wine splits."""
    errors: list[float] = []
    zero_steps = 0
    similar_violations = 0
    minimum_diagonals: list[float] = []

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
        X_train, X_test = _preprocess(
            X_train,
            X_test,
            mode=preprocessing,
        )
        factor, zero_count, violation_count, minimum_diagonal = _fit_factorized(
            X_train,
            y_train,
            seed=seed,
        )
        predictions = knn_predict(
            X_train @ factor,
            y_train,
            X_test @ factor,
            n_neighbors=N_NEIGHBORS,
        )
        errors.append(1.0 - accuracy_score(y_test, predictions))
        zero_steps += zero_count
        similar_violations += violation_count
        minimum_diagonals.append(minimum_diagonal)

    error_array = np.asarray(errors, dtype=np.float64)
    denominator = similar_violations if similar_violations > 0 else 1
    return FactorizedResult(
        error_mean=float(np.mean(error_array)),
        error_std=float(np.std(error_array, ddof=1)),
        zero_step_fraction=float(zero_steps / denominator),
        minimum_diagonal_mean=float(np.mean(minimum_diagonals)),
    )


def main() -> None:
    """Run the factorized strict-interior reference on Wine."""
    data = load_wine()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    print("RDML factorized strict-interior reference on Wine")
    print(f"Protocol: {N_NEIGHBORS}-NN, 50/50 split, {N_RUNS} fixed seeds")
    print(
        f"learning_rate={LEARNING_RATE}, max_iter={MAX_ITER}, "
        f"margin={MARGIN}, rho={RHO}, A0=I"
    )

    for mode in ("raw", "standardized"):
        result = _evaluate_mode(X, y, preprocessing=mode)
        print(
            f"{mode:>12}: error {100 * result.error_mean:5.2f}% ± "
            f"{100 * result.error_std:4.2f}; "
            f"zero similar steps {100 * result.zero_step_fraction:6.2f}%; "
            f"mean min diag(L) {result.minimum_diagonal_mean:.3e}"
        )


if __name__ == "__main__":
    main()
