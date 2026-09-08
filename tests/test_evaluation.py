"""Tests for NumPy-only evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from rdml import accuracy_score, knn_predict


def test_knn_predicts_separated_classes() -> None:
    """Nearest-neighbour voting recovers two well-separated classes."""
    X_train = np.array([[0.0], [0.2], [5.0], [5.2]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.1], [5.1]])

    predictions = knn_predict(X_train, y_train, X_test, n_neighbors=3)

    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_knn_tie_breaks_by_total_distance() -> None:
    """Equal vote counts prefer the class with the smaller distance sum."""
    predictions = knn_predict([[0.0], [2.0]], [0, 1], [[1.8]], n_neighbors=2)

    np.testing.assert_array_equal(predictions, np.array([1]))


def test_knn_exact_distance_tie_is_stable() -> None:
    """An exact distance and vote tie is resolved by neighbour order."""
    predictions = knn_predict([[-1.0], [1.0]], [7, 9], [[0.0]], n_neighbors=2)

    np.testing.assert_array_equal(predictions, np.array([7]))


def test_knn_preserves_string_label_dtype() -> None:
    """Predictions preserve non-numeric scalar labels."""
    predictions = knn_predict([[0.0], [3.0]], ["left", "right"], [[2.9]], n_neighbors=1)

    np.testing.assert_array_equal(predictions, np.array(["right"]))


def test_knn_validates_shapes_and_neighbour_count() -> None:
    """Malformed feature matrices and neighbour counts fail early."""
    with pytest.raises(ValueError, match="same number of features"):
        knn_predict([[0.0]], [0], [[0.0, 1.0]])
    with pytest.raises(ValueError, match="one label per sample"):
        knn_predict([[0.0], [1.0]], [0], [[0.5]])
    with pytest.raises(TypeError, match="integer"):
        knn_predict([[0.0]], [0], [[0.0]], n_neighbors=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        knn_predict([[0.0]], [0], [[0.0]], n_neighbors=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        knn_predict([[0.0]], [0], [[0.0]], n_neighbors=2)


def test_accuracy_score_computes_fraction_correct() -> None:
    """Accuracy is the exact fraction of matching labels."""
    assert accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)


def test_accuracy_score_validates_shapes() -> None:
    """Accuracy requires aligned, non-empty one-dimensional labels."""
    with pytest.raises(ValueError, match="same shape"):
        accuracy_score([0, 1], [0])
    with pytest.raises(ValueError, match="one-dimensional"):
        accuracy_score([[0, 1]], [0, 1])
    with pytest.raises(ValueError, match="at least one label"):
        accuracy_score([], [])
