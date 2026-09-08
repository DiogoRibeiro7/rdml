"""Tests for the mathematically faithful RDML reference implementation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from rdml import RDML, project_psd, squared_mahalanobis


def _mixed_dataset() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return a small dataset containing same-class and different-class pairs."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 0.0],
            [2.2, 0.1],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    return X, y


def test_project_psd_clips_negative_eigenvalues() -> None:
    """The PSD projection is symmetric and has no materially negative eigenvalues."""
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    projected = project_psd(matrix)

    np.testing.assert_allclose(projected, projected.T, atol=1e-12)
    assert np.linalg.eigvalsh(projected).min() >= -1e-12


def test_project_psd_rejects_invalid_inputs() -> None:
    """PSD projection validates shape and the eigenvalue floor."""
    with pytest.raises(ValueError, match="square"):
        project_psd(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        project_psd(np.eye(2), epsilon=-1.0)


def test_one_dissimilar_pair_matches_algorithm_one_update() -> None:
    """A violating dissimilar pair receives the paper's projected gradient update."""
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = np.array([0, 1])

    model = RDML(learning_rate=0.1, max_iter=1, random_state=7).fit(X, y)

    np.testing.assert_allclose(model.metric_, np.diag([0.4, 0.0]), atol=1e-12)
    assert model.squared_distance(X[0], X[1]) == pytest.approx(1.6)


def test_same_class_pair_is_correct_at_zero_metric() -> None:
    """Same-class pairs satisfy y_t * (b - d_A) > 0 at the zero metric."""
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 1])

    model = RDML(learning_rate=0.1, max_iter=10, random_state=2).fit(X, y)

    np.testing.assert_array_equal(model.metric_, np.zeros((2, 2)))


def test_fitted_metric_remains_positive_semidefinite() -> None:
    """Exact projection keeps the learned matrix inside the PSD cone."""
    X, y = _mixed_dataset()

    model = RDML(max_iter=250, random_state=13).fit(X, y)

    np.testing.assert_allclose(model.metric_, model.metric_.T, atol=1e-12)
    assert np.linalg.eigvalsh(model.metric_).min() >= -1e-10


def test_random_state_makes_training_reproducible() -> None:
    """Identical seeds produce identical pair sequences and learned metrics."""
    X, y = _mixed_dataset()

    first = RDML(max_iter=100, random_state=42).fit(X, y)
    second = RDML(max_iter=100, random_state=42).fit(X, y)

    np.testing.assert_allclose(first.metric_, second.metric_)


def test_transform_preserves_learned_squared_distance() -> None:
    """Euclidean distance after transform equals the learned Mahalanobis distance."""
    X, y = _mixed_dataset()
    model = RDML(max_iter=100, random_state=4).fit(X, y)

    transformed = model.transform(X)
    transformed_distance = float(np.sum((transformed[0] - transformed[2]) ** 2))

    assert transformed_distance == pytest.approx(model.squared_distance(X[0], X[2]))


def test_fit_transform_matches_separate_calls() -> None:
    """Convenience fit-transform agrees with an explicit fit followed by transform."""
    X, y = _mixed_dataset()
    direct = RDML(max_iter=50, random_state=3).fit_transform(X, y)
    separate_model = RDML(max_iter=50, random_state=3).fit(X, y)

    np.testing.assert_allclose(direct, separate_model.transform(X))


def test_squared_mahalanobis_validates_shapes() -> None:
    """Distance helper rejects incompatible vector and metric shapes."""
    with pytest.raises(ValueError, match="same shape"):
        squared_mahalanobis([0.0, 1.0], [0.0], np.eye(2))
    with pytest.raises(ValueError, match="metric shape"):
        squared_mahalanobis([0.0, 1.0], [1.0, 0.0], np.eye(3))


def test_estimator_validates_training_inputs_and_parameters() -> None:
    """The public estimator fails early on malformed data and invalid parameters."""
    with pytest.raises(ValueError, match="at least two samples"):
        RDML().fit([[0.0, 1.0]], [0])
    with pytest.raises(ValueError, match="same number of samples"):
        RDML().fit([[0.0], [1.0]], [0])
    with pytest.raises(ValueError, match="learning_rate"):
        RDML(learning_rate=0.0).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="margin"):
        RDML(margin=np.inf).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(TypeError, match="integer"):
        RDML(max_iter=1.5).fit([[0.0], [1.0]], [0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        RDML(max_iter=0).fit([[0.0], [1.0]], [0, 1])


def test_operations_require_fitted_estimator() -> None:
    """Transform and distance access cannot run before training."""
    model = RDML()

    with pytest.raises(RuntimeError, match="fitted"):
        model.transform([[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="fitted"):
        model.squared_distance([0.0], [1.0])


def test_transform_rejects_wrong_feature_count() -> None:
    """Transform enforces the dimensionality observed during fit."""
    X, y = _mixed_dataset()
    model = RDML(max_iter=10, random_state=1).fit(X, y)

    with pytest.raises(ValueError, match="different number of features"):
        model.transform([[0.0, 1.0, 2.0]])
