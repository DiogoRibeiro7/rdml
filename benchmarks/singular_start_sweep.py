"""Diagnose the singular-start ambiguity in the paper-efficient RDML update.

Theorem 6 uses an inverse of the current metric, while Algorithm 1 initializes
that metric at zero. This benchmark leaves the package estimator unchanged and
compares several explicitly named initialization conventions under the same
pair streams and 3-NN evaluation protocol used by the partial reproduction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris, load_wine

from rdml import RDML, accuracy_score, knn_predict, paper_step_size

FloatMatrix = NDArray[np.float64]
IntLabels = NDArray[np.int64]
DatasetName = Literal["iris", "wine"]

N_RUNS = 10
N_NEIGHBORS = 3
LEARNING_RATE = 0.1
MAX_ITER = 10_000
MARGIN = 1.0


@dataclass(frozen=True)
class StartVariant:
    """Named initialization convention for the paper-efficient update."""

    name: str
    identity_scale: float


@dataclass(frozen=True)
class VariantResult:
    """Repeated error and structural diagnostics for one variant."""

    error_mean: float
    error_std: float
    rank_mean: float
    zero_step_fraction: float


VARIANTS = (
    StartVariant("paper-zero", 0.0),
    StartVariant("paper-0.1I", 0.1),
    StartVariant("paper-I", 1.0),
    StartVariant("paper-10I", 10.0),
)


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


def _components(metric: FloatMatrix) -> FloatMatrix:
    """Return a Euclidean transform corresponding to a PSD metric."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    components = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    return np.asarray(components, dtype=np.float64)


def _numerical_rank(metric: FloatMatrix) -> int:
    """Return the same scale-aware rank convention used by the estimator."""
    eigenvalues = np.asarray(np.linalg.eigvalsh(0.5 * (metric + metric.T)), dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * eigenvalues.size * scale
    return int(np.count_nonzero(eigenvalues > tolerance))


def _fit_paper_variant(
    X: FloatMatrix,
    y: IntLabels,
    *,
    seed: int,
    identity_scale: float,
) -> tuple[FloatMatrix, int, int]:
    """Fit the public Theorem 6 step from a chosen identity initialization.

    This deliberately reuses :func:`rdml.paper_step_size`; only the initial
    metric differs from the stable estimator. A scale of zero reproduces the
    package's current singular-safe paper path.
    """
    n_samples, n_features = X.shape
    metric = np.eye(n_features, dtype=np.float64) * identity_scale
    rng = np.random.default_rng(seed)
    similar_violations = 0
    zero_similar_steps = 0

    for _ in range(MAX_ITER):
        first, second = rng.choice(n_samples, size=2, replace=False)
        difference = X[first] - X[second]
        pair_label = 1.0 if y[first] == y[second] else -1.0
        distance = float(difference @ metric @ difference)

        if pair_label * (MARGIN - distance) > 0.0:
            continue

        step = paper_step_size(
            metric,
            difference,
            pair_label,
            LEARNING_RATE,
        )
        if pair_label == 1.0:
            similar_violations += 1
            if step == 0.0:
                zero_similar_steps += 1

        metric = metric - step * pair_label * np.outer(difference, difference)
        metric = np.asarray(0.5 * (metric + metric.T), dtype=np.float64)

    return metric, zero_similar_steps, similar_violations


def _classification_error(
    metric: FloatMatrix,
    X_train: FloatMatrix,
    y_train: IntLabels,
    X_test: FloatMatrix,
    y_test: IntLabels,
) -> float:
    """Evaluate one metric with deterministic 3-NN."""
    components = _components(metric)
    predictions = knn_predict(
        X_train @ components,
        y_train,
        X_test @ components,
        n_neighbors=N_NEIGHBORS,
    )
    return 1.0 - accuracy_score(y_test, predictions)


def _evaluate_exact(X: FloatMatrix, y: IntLabels) -> VariantResult:
    """Evaluate the exact projected Algorithm 1 reference."""
    errors: list[float] = []
    ranks: list[int] = []

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
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
        ranks.append(model.diagnostics_.final_rank)

    error_array = np.asarray(errors, dtype=np.float64)
    return VariantResult(
        error_mean=float(np.mean(error_array)),
        error_std=float(np.std(error_array, ddof=1)),
        rank_mean=float(np.mean(ranks)),
        zero_step_fraction=0.0,
    )


def _evaluate_variant(
    X: FloatMatrix,
    y: IntLabels,
    variant: StartVariant,
) -> VariantResult:
    """Evaluate one singular-start convention over the ten fixed seeds."""
    errors: list[float] = []
    ranks: list[int] = []
    zero_steps = 0
    similar_violations = 0

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)
        metric, zero_count, violation_count = _fit_paper_variant(
            X_train,
            y_train,
            seed=seed,
            identity_scale=variant.identity_scale,
        )
        errors.append(_classification_error(metric, X_train, y_train, X_test, y_test))
        ranks.append(_numerical_rank(metric))
        zero_steps += zero_count
        similar_violations += violation_count

    error_array = np.asarray(errors, dtype=np.float64)
    zero_fraction = zero_steps / similar_violations if similar_violations else 0.0
    return VariantResult(
        error_mean=float(np.mean(error_array)),
        error_std=float(np.std(error_array, ddof=1)),
        rank_mean=float(np.mean(ranks)),
        zero_step_fraction=float(zero_fraction),
    )


def _dataset(name: DatasetName) -> tuple[FloatMatrix, IntLabels]:
    """Load one bundled UCI dataset without a network request."""
    data = load_iris() if name == "iris" else load_wine()
    return (
        np.asarray(data.data, dtype=np.float64),
        np.asarray(data.target, dtype=np.int64),
    )


def _print_dataset(name: DatasetName) -> None:
    """Run and print the complete start-convention comparison."""
    X, y = _dataset(name)
    print(f"\n{name.title()}")
    print("-" * len(name))

    exact = _evaluate_exact(X, y)
    print(
        f"{'exact':>12}: error {100 * exact.error_mean:5.2f}% ± "
        f"{100 * exact.error_std:4.2f}; rank {exact.rank_mean:4.1f}"
    )

    for variant in VARIANTS:
        result = _evaluate_variant(X, y, variant)
        print(
            f"{variant.name:>12}: error {100 * result.error_mean:5.2f}% ± "
            f"{100 * result.error_std:4.2f}; rank {result.rank_mean:4.1f}; "
            f"zero similar steps {100 * result.zero_step_fraction:6.2f}%"
        )


def main() -> None:
    """Run the singular-start sweep without modifying the stable estimator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("iris", "wine", "all"),
        default="all",
        help="Dataset subset to evaluate.",
    )
    args = parser.parse_args()

    print("RDML Theorem 6 singular-start sweep")
    print(f"Protocol: {N_NEIGHBORS}-NN, 50/50 split, {N_RUNS} fixed seeds")
    print(f"learning_rate={LEARNING_RATE}, max_iter={MAX_ITER}, margin={MARGIN}")
    print("Identity starts are diagnostics, not proposed defaults.")

    if args.dataset in ("iris", "all"):
        _print_dataset("iris")
    if args.dataset in ("wine", "all"):
        _print_dataset("wine")


if __name__ == "__main__":
    main()
