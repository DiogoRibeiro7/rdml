"""Reference sweep for strict-interior RDML steps on Wine.

This benchmark is intentionally outside the package API. It tests the
mathematical convention

    alpha_t = min(lambda, rho / (v.T @ A^{-1} @ v)),  0 < rho < 1,

from A_0 = I, using a direct linear solve as a numerical reference. It also
compares raw features with train-only standardization to expose the interaction
between conditioning and the published Euclidean baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_wine

from rdml import RDML, accuracy_score, knn_predict

FloatMatrix = NDArray[np.float64]
IntLabels = NDArray[np.int64]
Preprocessing = Literal["raw", "standardized"]

N_RUNS = 10
N_NEIGHBORS = 3
LEARNING_RATE = 0.1
MAX_ITER = 10_000
MARGIN = 1.0
RHOS = (0.01, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class VariantResult:
    """Repeated error and structural diagnostics for one interior cap."""

    error_mean: float
    error_std: float
    rank_mean: float
    minimum_eigenvalue_mean: float
    zero_step_fraction: float
    capped_step_fraction: float


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


def _components(metric: FloatMatrix) -> FloatMatrix:
    """Return Euclidean coordinates for a symmetric PSD metric."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    return np.asarray(
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0)),
        dtype=np.float64,
    )


def _numerical_rank(metric: FloatMatrix) -> int:
    """Return a scale-aware numerical rank."""
    eigenvalues = np.asarray(
        np.linalg.eigvalsh(0.5 * (metric + metric.T)),
        dtype=np.float64,
    )
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * eigenvalues.size * scale
    return int(np.count_nonzero(eigenvalues > tolerance))


def _error(
    metric: FloatMatrix,
    X_train: FloatMatrix,
    y_train: IntLabels,
    X_test: FloatMatrix,
    y_test: IntLabels,
) -> float:
    """Evaluate a learned metric with deterministic 3-NN."""
    components = _components(metric)
    predictions = knn_predict(
        X_train @ components,
        y_train,
        X_test @ components,
        n_neighbors=N_NEIGHBORS,
    )
    return 1.0 - accuracy_score(y_test, predictions)


def _fit_interior_reference(
    X: FloatMatrix,
    y: IntLabels,
    *,
    seed: int,
    rho: float,
) -> tuple[FloatMatrix, int, int, int]:
    """Fit a strict-interior reference using direct solves.

    Direct solves are deliberate here: this benchmark tests the mathematical
    convention separately from any scalable inverse-maintenance algorithm.
    """
    n_samples, n_features = X.shape
    metric = np.eye(n_features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    similar_violations = 0
    zero_steps = 0
    capped_steps = 0

    for _ in range(MAX_ITER):
        first, second = rng.choice(n_samples, size=2, replace=False)
        difference = X[first] - X[second]
        pair_label = 1.0 if y[first] == y[second] else -1.0
        distance = float(difference @ metric @ difference)

        if pair_label * (MARGIN - distance) > 0.0:
            continue

        if pair_label == -1.0:
            step = LEARNING_RATE
        else:
            similar_violations += 1
            try:
                solution = np.linalg.solve(metric, difference)
                inverse_quadratic = float(difference @ solution)
            except np.linalg.LinAlgError:
                inverse_quadratic = float("inf")

            if not np.isfinite(inverse_quadratic) or inverse_quadratic <= 0.0:
                step = 0.0
                zero_steps += 1
            else:
                cap = rho / inverse_quadratic
                step = min(LEARNING_RATE, cap)
                if step < LEARNING_RATE:
                    capped_steps += 1
                if step == 0.0:
                    zero_steps += 1

        metric = metric - step * pair_label * np.outer(difference, difference)
        metric = np.asarray(0.5 * (metric + metric.T), dtype=np.float64)

    return metric, zero_steps, capped_steps, similar_violations


def _evaluate_interior(
    X: FloatMatrix,
    y: IntLabels,
    *,
    rho: float,
    preprocessing: Preprocessing,
) -> VariantResult:
    """Evaluate one strict-interior cap over the ten fixed splits."""
    errors: list[float] = []
    ranks: list[int] = []
    minimum_eigenvalues: list[float] = []
    zero_steps = 0
    capped_steps = 0
    similar_violations = 0

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
        X_train, X_test = _preprocess(
            X_train,
            X_test,
            mode=preprocessing,
        )
        metric, zero_count, capped_count, violation_count = _fit_interior_reference(
            X_train,
            y_train,
            seed=seed,
            rho=rho,
        )
        errors.append(_error(metric, X_train, y_train, X_test, y_test))
        ranks.append(_numerical_rank(metric))
        minimum_eigenvalues.append(
            float(np.min(np.linalg.eigvalsh(metric)))
        )
        zero_steps += zero_count
        capped_steps += capped_count
        similar_violations += violation_count

    error_array = np.asarray(errors, dtype=np.float64)
    denominator = similar_violations if similar_violations > 0 else 1
    return VariantResult(
        error_mean=float(np.mean(error_array)),
        error_std=float(np.std(error_array, ddof=1)),
        rank_mean=float(np.mean(ranks)),
        minimum_eigenvalue_mean=float(np.mean(minimum_eigenvalues)),
        zero_step_fraction=float(zero_steps / denominator),
        capped_step_fraction=float(capped_steps / denominator),
    )


def _euclidean_error(
    X: FloatMatrix,
    y: IntLabels,
    *,
    preprocessing: Preprocessing,
) -> tuple[float, float]:
    """Return repeated Euclidean 3-NN error for one preprocessing convention."""
    errors: list[float] = []

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
        X_train, X_test = _preprocess(
            X_train,
            X_test,
            mode=preprocessing,
        )
        predictions = knn_predict(
            X_train,
            y_train,
            X_test,
            n_neighbors=N_NEIGHBORS,
        )
        errors.append(1.0 - accuracy_score(y_test, predictions))

    error_array = np.asarray(errors, dtype=np.float64)
    return float(np.mean(error_array)), float(np.std(error_array, ddof=1))


def _exact_error(
    X: FloatMatrix,
    y: IntLabels,
    *,
    preprocessing: Preprocessing,
) -> tuple[float, float]:
    """Return repeated exact-projection RDML error."""
    errors: list[float] = []

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
        X_train, X_test = _preprocess(
            X_train,
            X_test,
            mode=preprocessing,
        )
        model = RDML(
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            margin=MARGIN,
            random_state=seed,
            update_method="exact",
        ).fit(X_train, y_train)
        predictions = knn_predict(
            model.transform(X_train),
            y_train,
            model.transform(X_test),
            n_neighbors=N_NEIGHBORS,
        )
        errors.append(1.0 - accuracy_score(y_test, predictions))

    error_array = np.asarray(errors, dtype=np.float64)
    return float(np.mean(error_array)), float(np.std(error_array, ddof=1))


def _print_mode(
    X: FloatMatrix,
    y: IntLabels,
    preprocessing: Preprocessing,
) -> None:
    """Print baselines and the complete rho sweep for one preprocessing mode."""
    print(f"\n{preprocessing}")
    print("-" * len(preprocessing))

    euclidean_mean, euclidean_std = _euclidean_error(
        X,
        y,
        preprocessing=preprocessing,
    )
    exact_mean, exact_std = _exact_error(
        X,
        y,
        preprocessing=preprocessing,
    )
    print(
        f"{'euclidean':>12}: error {100 * euclidean_mean:5.2f}% ± "
        f"{100 * euclidean_std:4.2f}"
    )
    print(
        f"{'exact':>12}: error {100 * exact_mean:5.2f}% ± "
        f"{100 * exact_std:4.2f}"
    )

    for rho in RHOS:
        result = _evaluate_interior(
            X,
            y,
            rho=rho,
            preprocessing=preprocessing,
        )
        print(
            f"rho={rho:0.2f}: error {100 * result.error_mean:5.2f}% ± "
            f"{100 * result.error_std:4.2f}; rank {result.rank_mean:4.1f}; "
            f"min eig {result.minimum_eigenvalue_mean:.3e}; "
            f"zero {100 * result.zero_step_fraction:6.2f}%; "
            f"capped {100 * result.capped_step_fraction:6.2f}%"
        )


def main() -> None:
    """Run the strict-interior reference sweep on Wine."""
    data = load_wine()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    print("RDML strict-interior reference sweep on Wine")
    print(f"Protocol: {N_NEIGHBORS}-NN, 50/50 split, {N_RUNS} fixed seeds")
    print(
        f"learning_rate={LEARNING_RATE}, max_iter={MAX_ITER}, "
        f"margin={MARGIN}, A0=I"
    )
    print("Interior variants are research diagnostics, not proposed defaults.")

    _print_mode(X, y, "raw")
    _print_mode(X, y, "standardized")


if __name__ == "__main__":
    main()
