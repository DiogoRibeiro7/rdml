"""Dependency-light evaluation utilities for learned metric spaces."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[Any]


def _as_float_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    """Return ``values`` as a finite two-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_labels(values: ArrayLike, *, name: str, n_samples: int | None = None) -> LabelArray:
    """Return labels as a non-empty one-dimensional array."""
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if labels.size == 0:
        raise ValueError(f"{name} must contain at least one label.")
    if n_samples is not None and labels.shape[0] != n_samples:
        raise ValueError(f"{name} must contain one label per sample.")
    return labels


def _validate_n_neighbors(value: int, *, n_train: int) -> int:
    """Validate the requested number of neighbours."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_neighbors must be an integer.")
    if value <= 0:
        raise ValueError("n_neighbors must be greater than zero.")
    if value > n_train:
        raise ValueError("n_neighbors cannot exceed the number of training samples.")
    return value


def _same_label(first: Any, second: Any) -> bool:
    """Return whether two scalar labels compare equal."""
    comparison = first == second
    if isinstance(comparison, (bool, np.bool_)):
        return bool(comparison)
    return False


def _majority_label(labels: LabelArray, distances: FloatArray) -> Any:
    """Choose a majority label with deterministic tie-breaking.

    Ties are broken first by the smallest sum of squared neighbour distances
    and then by the earliest neighbour rank.
    """
    candidates: list[Any] = []
    counts: list[int] = []
    distance_sums: list[float] = []
    first_ranks: list[int] = []

    for rank, (label, distance) in enumerate(zip(labels, distances, strict=True)):
        for index, candidate in enumerate(candidates):
            if _same_label(label, candidate):
                counts[index] += 1
                distance_sums[index] += float(distance)
                break
        else:
            candidates.append(label)
            counts.append(1)
            distance_sums.append(float(distance))
            first_ranks.append(rank)

    best_index = min(
        range(len(candidates)),
        key=lambda index: (-counts[index], distance_sums[index], first_ranks[index]),
    )
    return candidates[best_index]


def knn_predict(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    *,
    n_neighbors: int = 3,
) -> LabelArray:
    """Predict labels with deterministic Euclidean k-nearest neighbours."""
    train = _as_float_matrix(X_train, name="X_train")
    test = _as_float_matrix(X_test, name="X_test")
    labels = _as_labels(y_train, name="y_train", n_samples=train.shape[0])
    if train.shape[1] != test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")
    resolved_neighbors = _validate_n_neighbors(n_neighbors, n_train=train.shape[0])

    predictions: list[Any] = []
    for query in test:
        squared_distances = np.sum((train - query) ** 2, axis=1, dtype=np.float64)
        neighbour_indices = np.argsort(squared_distances, kind="stable")[:resolved_neighbors]
        predictions.append(
            _majority_label(labels[neighbour_indices], squared_distances[neighbour_indices])
        )

    return np.asarray(predictions, dtype=labels.dtype)


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of exactly matching labels."""
    truth = _as_labels(y_true, name="y_true")
    predictions = _as_labels(y_pred, name="y_pred")
    if truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(truth == predictions))
