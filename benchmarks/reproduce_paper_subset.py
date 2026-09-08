"""Partial reproduction of Jin, Wang, and Zhou (2009), Experiment I.

This script reproduces the published evaluation structure on the Iris and Wine
UCI datasets: 3-NN, a random 50/50 train/test split, and 10 repeated runs.

It is intentionally labelled a partial reproduction rather than an exact
replication. The paper does not publish the random seeds, sampled pair stream,
or all tuning details needed to recreate Table 1 bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris, load_wine

from rdml import RDML, accuracy_score, knn_predict

FloatMatrix = NDArray[np.float64]
IntLabels = NDArray[np.int64]

N_RUNS = 10
N_NEIGHBORS = 3
LEARNING_RATE = 0.1
MAX_ITER = 10_000
MARGIN = 1.0


@dataclass(frozen=True)
class PaperReference:
    """Published Table 1 error rates for one dataset."""

    euclidean_mean: float
    euclidean_std: float
    online_reg_mean: float
    online_reg_std: float


@dataclass(frozen=True)
class ReproductionResult:
    """Mean and sample standard deviation of repeated classification errors."""

    euclidean_mean: float
    euclidean_std: float
    rdml_mean: float
    rdml_std: float


def _split_half(
    X: FloatMatrix,
    y: IntLabels,
    *,
    seed: int,
) -> tuple[FloatMatrix, IntLabels, FloatMatrix, IntLabels]:
    """Return one deterministic 50/50 random split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    midpoint = X.shape[0] // 2
    train_idx = order[:midpoint]
    test_idx = order[midpoint:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _classification_error(y_true: IntLabels, y_pred: NDArray[np.generic]) -> float:
    """Return classification error as a fraction in [0, 1]."""
    return 1.0 - accuracy_score(y_true, y_pred)


def evaluate_dataset(X: FloatMatrix, y: IntLabels) -> ReproductionResult:
    """Evaluate Euclidean and paper-update RDML under the published split protocol."""
    euclidean_errors: list[float] = []
    rdml_errors: list[float] = []

    for seed in range(N_RUNS):
        X_train, y_train, X_test, y_test = _split_half(X, y, seed=seed)

        euclidean_pred = knn_predict(
            X_train,
            y_train,
            X_test,
            n_neighbors=N_NEIGHBORS,
        )
        euclidean_errors.append(_classification_error(y_test, euclidean_pred))

        model = RDML(
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            margin=MARGIN,
            random_state=seed,
            update_method="paper",
        ).fit(X_train, y_train)
        transformed_train = model.transform(X_train)
        transformed_test = model.transform(X_test)
        rdml_pred = knn_predict(
            transformed_train,
            y_train,
            transformed_test,
            n_neighbors=N_NEIGHBORS,
        )
        rdml_errors.append(_classification_error(y_test, rdml_pred))

    euclidean = np.asarray(euclidean_errors, dtype=np.float64)
    rdml = np.asarray(rdml_errors, dtype=np.float64)
    return ReproductionResult(
        euclidean_mean=float(np.mean(euclidean)),
        euclidean_std=float(np.std(euclidean, ddof=1)),
        rdml_mean=float(np.mean(rdml)),
        rdml_std=float(np.std(rdml, ddof=1)),
    )


def _print_result(
    name: str,
    result: ReproductionResult,
    reference: PaperReference,
) -> None:
    """Print reproduced and published error rates in percentage points."""
    print(f"\n{name}")
    print("-" * len(name))
    print(
        "This implementation: "
        f"Euclidean {100 * result.euclidean_mean:.1f}% ± {100 * result.euclidean_std:.1f}; "
        f"RDML {100 * result.rdml_mean:.1f}% ± {100 * result.rdml_std:.1f}"
    )
    print(
        "Paper Table 1:       "
        f"Euclidean {reference.euclidean_mean:.1f}% ± {reference.euclidean_std:.1f}; "
        f"online-reg {reference.online_reg_mean:.1f}% ± {reference.online_reg_std:.1f}"
    )


def main() -> None:
    """Run the protocol-aligned partial reproduction."""
    iris = load_iris()
    wine = load_wine()

    datasets = [
        (
            "Iris",
            np.asarray(iris.data, dtype=np.float64),
            np.asarray(iris.target, dtype=np.int64),
            PaperReference(4.0, 1.7, 3.2, 1.3),
        ),
        (
            "Wine",
            np.asarray(wine.data, dtype=np.float64),
            np.asarray(wine.target, dtype=np.int64),
            PaperReference(31.9, 2.8, 1.8, 1.1),
        ),
    ]

    print("RDML partial reproduction of Jin, Wang, and Zhou (2009), Experiment I")
    print(f"Protocol: {N_NEIGHBORS}-NN, 50/50 random split, {N_RUNS} runs")
    print(
        "RDML configuration: "
        f"paper update, learning_rate={LEARNING_RATE}, max_iter={MAX_ITER}, margin={MARGIN}"
    )
    print("No feature standardization is applied.")
    print("Published values are context, not exact-regression targets.")

    for name, X, y, reference in datasets:
        _print_result(name, evaluate_dataset(X, y), reference)


if __name__ == "__main__":
    main()
