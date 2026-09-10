"""Core Regularized Distance Metric Learning implementation.

The implementation follows Algorithm 1 from Jin, Wang, and Zhou (2009). The
exact positive-semidefinite projection remains the correctness reference, while
the paper's efficient adaptive rank-one update is available as an opt-in method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
LabelArray: TypeAlias = NDArray[Any]
RandomState: TypeAlias = int | np.random.Generator | None
UpdateMethod: TypeAlias = Literal["exact", "paper"]


@dataclass(frozen=True)
class FitDiagnostics:
    """Summary of the update stream observed during one RDML fit."""

    sampled_pairs: int
    correct_pairs: int
    dissimilar_updates: int
    similar_violations: int
    similar_positive_steps: int
    similar_zero_steps: int
    final_rank: int
    minimum_eigenvalue: float


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


def _as_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    """Return ``values`` as a finite one-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
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


def _validate_positive_int(value: int, *, name: str) -> int:
    """Validate a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _validate_update_method(value: str) -> UpdateMethod:
    """Validate and narrow the requested update method."""
    if value == "exact":
        return "exact"
    if value == "paper":
        return "paper"
    raise ValueError("update_method must be either 'exact' or 'paper'.")


def _numerical_rank(eigenvalues: FloatArray) -> int:
    """Return a scale-aware numerical rank for a PSD metric."""
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * eigenvalues.size * scale
    return int(np.count_nonzero(eigenvalues > tolerance))


def project_psd(matrix: ArrayLike, *, epsilon: float = 0.0) -> FloatArray:
    """Project a square matrix onto the positive semi-definite cone."""
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


def _inverse_quadratic_form_cg(
    metric: FloatArray,
    difference: FloatArray,
    *,
    tolerance: float,
    max_iter: int,
) -> float:
    """Approximate ``difference.T @ metric^-1 @ difference`` using CG.

    A singular PSD system is valid when ``difference`` lies in the range of
    ``metric``. If CG encounters a null-space direction before convergence, the
    associated quadratic maximization is unbounded and ``inf`` is returned.
    """
    norm_difference = float(np.linalg.norm(difference))
    if norm_difference == 0.0:
        return 0.0

    solution = np.zeros_like(difference, dtype=np.float64)
    residual = difference.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    target_sq = (tolerance * max(1.0, norm_difference)) ** 2
    matrix_scale = max(1.0, float(np.linalg.norm(metric, ord="fro")))
    machine_epsilon = np.finfo(np.float64).eps

    for _ in range(max_iter):
        metric_direction = np.asarray(metric @ direction, dtype=np.float64)
        direction_sq = float(direction @ direction)
        curvature = float(direction @ metric_direction)
        curvature_floor = machine_epsilon * matrix_scale * max(1.0, direction_sq)
        if curvature <= curvature_floor:
            return float("inf")

        step = residual_sq / curvature
        solution = solution + step * direction
        residual = residual - step * metric_direction
        next_residual_sq = float(residual @ residual)
        if next_residual_sq <= target_sq:
            quadratic_form = float(difference @ solution)
            if quadratic_form >= 0.0:
                return quadratic_form
            return float("inf")

        direction = residual + (next_residual_sq / residual_sq) * direction
        residual_sq = next_residual_sq

    return float("inf")


def _paper_step_size_validated(
    metric: FloatArray,
    difference: FloatArray,
    pair_label: float,
    learning_rate: float,
    *,
    cg_tolerance: float,
    cg_max_iter: int,
) -> float:
    """Return the feasible paper step after inputs have been validated."""
    if pair_label == -1.0:
        return learning_rate

    inverse_quadratic = _inverse_quadratic_form_cg(
        metric,
        difference,
        tolerance=cg_tolerance,
        max_iter=cg_max_iter,
    )
    if not np.isfinite(inverse_quadratic):
        return 0.0
    if inverse_quadratic == 0.0:
        return learning_rate
    return min(learning_rate, 1.0 / inverse_quadratic)


def paper_step_size(
    metric: ArrayLike,
    difference: ArrayLike,
    pair_label: float,
    learning_rate: float,
    *,
    cg_tolerance: float = 1e-8,
    cg_max_iter: int | None = None,
) -> float:
    """Return Theorem 6's PSD-preserving adaptive learning rate.

    For dissimilar pairs (``pair_label == -1``), Theorem 6 keeps the full
    learning rate. For similar pairs, the step is capped by the reciprocal of
    ``difference.T @ metric^-1 @ difference``. The inverse quadratic form is
    evaluated with conjugate gradient, as proposed in the paper.

    If a singular PSD system is inconsistent, no positive rank-one subtraction
    can preserve positive semidefiniteness, so the conservative feasible step is
    zero.
    """
    metric_array = _as_float_matrix(metric, name="metric")
    if metric_array.shape[0] != metric_array.shape[1]:
        raise ValueError("metric must be square.")
    difference_array = _as_float_vector(difference, name="difference")
    if difference_array.size != metric_array.shape[0]:
        raise ValueError("difference size must match the metric dimension.")
    if pair_label not in (-1.0, 1.0):
        raise ValueError("pair_label must be either -1.0 or 1.0.")

    resolved_learning_rate = _validate_positive_float(learning_rate, name="learning_rate")
    resolved_tolerance = _validate_positive_float(cg_tolerance, name="cg_tolerance")
    resolved_max_iter = (
        2 * metric_array.shape[0]
        if cg_max_iter is None
        else _validate_positive_int(cg_max_iter, name="cg_max_iter")
    )

    return _paper_step_size_validated(
        metric_array,
        difference_array,
        pair_label,
        resolved_learning_rate,
        cg_tolerance=resolved_tolerance,
        cg_max_iter=resolved_max_iter,
    )


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
    """Online Regularized Distance Metric Learning estimator."""

    metric_: FloatArray
    components_: FloatArray
    n_features_in_: int
    diagnostics_: FitDiagnostics

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 1_000,
        margin: float = 1.0,
        random_state: RandomState = None,
        update_method: UpdateMethod = "exact",
        cg_tolerance: float = 1e-8,
        cg_max_iter: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.margin = margin
        self.random_state = random_state
        self.update_method = update_method
        self.cg_tolerance = cg_tolerance
        self.cg_max_iter = cg_max_iter

    def fit(self, X: ArrayLike, y: ArrayLike) -> RDML:
        """Learn a PSD distance metric and expose update-stream diagnostics."""
        features = _as_float_matrix(X, name="X")
        if features.shape[0] < 2:
            raise ValueError("X must contain at least two samples.")
        labels = _as_labels(y, n_samples=features.shape[0])

        learning_rate = _validate_positive_float(self.learning_rate, name="learning_rate")
        margin = _validate_positive_float(self.margin, name="margin")
        max_iter = _validate_positive_int(self.max_iter, name="max_iter")
        update_method = _validate_update_method(self.update_method)
        cg_tolerance = _validate_positive_float(self.cg_tolerance, name="cg_tolerance")

        n_samples, n_features = features.shape
        cg_max_iter = (
            2 * n_features
            if self.cg_max_iter is None
            else _validate_positive_int(self.cg_max_iter, name="cg_max_iter")
        )
        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        metric = np.zeros((n_features, n_features), dtype=np.float64)
        correct_pairs = 0
        dissimilar_updates = 0
        similar_violations = 0
        similar_positive_steps = 0
        similar_zero_steps = 0

        for _ in range(max_iter):
            first, second = rng.choice(n_samples, size=2, replace=False)
            difference = features[first] - features[second]
            pair_label = 1.0 if labels[first] == labels[second] else -1.0
            distance = float(difference @ metric @ difference)

            if pair_label * (margin - distance) > 0.0:
                correct_pairs += 1
                continue

            if pair_label == -1.0:
                dissimilar_updates += 1
            else:
                similar_violations += 1

            if update_method == "exact":
                candidate = metric - learning_rate * pair_label * np.outer(difference, difference)
                metric = project_psd(candidate)
                if pair_label == 1.0:
                    similar_positive_steps += 1
            else:
                adaptive_step = _paper_step_size_validated(
                    metric,
                    difference,
                    pair_label,
                    learning_rate,
                    cg_tolerance=cg_tolerance,
                    cg_max_iter=cg_max_iter,
                )
                if pair_label == 1.0:
                    if adaptive_step > 0.0:
                        similar_positive_steps += 1
                    else:
                        similar_zero_steps += 1
                metric = metric - adaptive_step * pair_label * np.outer(difference, difference)
                metric = np.asarray(0.5 * (metric + metric.T), dtype=np.float64)

        eigenvalues = np.asarray(np.linalg.eigvalsh(metric), dtype=np.float64)
        self.metric_ = metric
        self.n_features_in_ = n_features
        self.components_ = self._metric_components(metric)
        self.diagnostics_ = FitDiagnostics(
            sampled_pairs=max_iter,
            correct_pairs=correct_pairs,
            dissimilar_updates=dissimilar_updates,
            similar_violations=similar_violations,
            similar_positive_steps=similar_positive_steps,
            similar_zero_steps=similar_zero_steps,
            final_rank=_numerical_rank(eigenvalues),
            minimum_eigenvalue=float(np.min(eigenvalues)),
        )
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
