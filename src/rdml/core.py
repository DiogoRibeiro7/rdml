"""Core Regularized Distance Metric Learning implementation.

The implementation follows Algorithm 1 from Jin, Wang, and Zhou (2009).  The
first modernized implementation deliberately uses the exact positive
semi-definite projection from the algorithm as a correctness reference.  The
paper's faster approximate projection can then be implemented and tested
against this baseline.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[Any]
RandomState: TypeAlias = int | np.random.Generator | None


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


def _as_labels(values: ArrayLike, *, n_samples: int) -> LabelArray:
    """Return labels as a one-dimensional array aligned with the samples."""
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if labels.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    return labels


def _validate_positive_float(value: float, *, name: str) -> float:
    """Validate a finite strictly positive floating-point parameter."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than zero.")
    return numeric


def project_psd(matrix: ArrayLike, *, epsilon: float = 0.0) -> FloatArray:
    """Project a square matrix onto the positive semi-definite cone.

    Parameters
    ----------
    matrix:
        Matrix to project.  Any numerical asymmetry is removed before the
        eigendecomposition.
    epsilon:
        Minimum allowed eigenvalue after projection.  ``0`` gives the nearest
        positive semi-definite matrix under eigenvalue clipping.

    Returns
    -------
    numpy.ndarray
        Symmetric positive semi-definite matrix.
    """
    array = _as_float_matrix(matrix, name="matrix")
    if array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be a finite non-negative value.")

    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(epsilon))
    projected = (eigenvectors * clipped) @ eigenvectors.T
    return np.asarray(0.5 * (projected + projected.T), dtype=np.float64)


def squared_mahalanobis(x: ArrayLike, y: ArrayLike, metric: ArrayLike) -> float:
    """Return the squared Mahalanobis distance induced by ``metric``."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    metric_array = _as_float_matrix(metric, name="metric")

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional vectors.")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape.")
    if metric_array.shape != (x_array.size, x_array.size):
        raise ValueError("metric shape must match the dimensionality of x and y.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must contain only finite values.")

    difference = x_array - y_array
    return float(difference @ metric_array @ difference)


class RDML:
    """Online Regularized Distance Metric Learning estimator.

    This estimator implements Algorithm 1 from Jin, Wang, and Zhou (2009) using
    exact projection onto the positive semi-definite cone after every violating
    pair.  It intentionally prioritizes a transparent mathematical reference
    implementation over the paper's faster approximate projection.

    Parameters
    ----------
    learning_rate:
        Online update step size :math:`\\lambda`.
    max_iter:
        Number of randomly sampled training pairs.
    margin:
        Classification margin :math:`b` in the hinge-loss formulation.
    random_state:
        Seed or NumPy generator used to sample training pairs.
    """

    metric_: FloatArray
    components_: FloatArray
    n_features_in_: int

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 1_000,
        margin: float = 1.0,
        random_state: RandomState = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.margin = margin
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> RDML:
        """Learn a positive semi-definite distance metric from labelled samples."""
        features = _as_float_matrix(X, name="X")
        if features.shape[0] < 2:
            raise ValueError("X must contain at least two samples.")
        labels = _as_labels(y, n_samples=features.shape[0])

        learning_rate = _validate_positive_float(self.learning_rate, name="learning_rate")
        margin = _validate_positive_float(self.margin, name="margin")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")

        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )

        n_samples, n_features = features.shape
        metric = np.zeros((n_features, n_features), dtype=np.float64)

        for _ in range(self.max_iter):
            first, second = rng.choice(n_samples, size=2, replace=False)
            difference = features[first] - features[second]
            pair_label = 1.0 if labels[first] == labels[second] else -1.0
            distance = float(difference @ metric @ difference)

            # Algorithm 1: a pair is correct when y_t * (b - d_A) > 0.
            if pair_label * (margin - distance) > 0.0:
                continue

            candidate = metric - learning_rate * pair_label * np.outer(difference, difference)
            metric = project_psd(candidate)

        self.metric_ = metric
        self.n_features_in_ = n_features
        self.components_ = self._metric_components(metric)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        """Map samples to a Euclidean space representing the learned metric."""
        self._require_fitted()
        features = _as_float_matrix(X, name="X")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data.")
        return features @ self.components_

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> FloatArray:
        """Fit the estimator and transform the training samples."""
        return self.fit(X, y).transform(X)

    def squared_distance(self, x: ArrayLike, y: ArrayLike) -> float:
        """Return the squared distance between two samples under the learned metric."""
        self._require_fitted()
        return squared_mahalanobis(x, y, self.metric_)

    @staticmethod
    def _metric_components(metric: FloatArray) -> FloatArray:
        """Return a linear transform whose Euclidean metric equals ``metric``."""
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        safe_eigenvalues = np.maximum(eigenvalues, 0.0)
        components = eigenvectors * np.sqrt(safe_eigenvalues)
        return np.asarray(components, dtype=np.float64)

    def _require_fitted(self) -> None:
        """Raise when a fitted attribute is requested before ``fit``."""
        if not hasattr(self, "metric_"):
            raise RuntimeError("RDML must be fitted before this operation.")
