"""Tests for observable RDML update-stream diagnostics."""

from __future__ import annotations

import numpy as np

from rdml import RDML, FitDiagnostics


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a small labelled dataset exercising both pair types."""
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


def test_exact_fit_diagnostics_partition_sampled_pairs() -> None:
    """Exact-mode counters form a complete partition of sampled pairs."""
    X, y = _dataset()
    model = RDML(max_iter=250, random_state=13, update_method="exact").fit(X, y)
    diagnostics = model.diagnostics_

    assert isinstance(diagnostics, FitDiagnostics)
    assert diagnostics.sampled_pairs == 250
    assert (
        diagnostics.correct_pairs
        + diagnostics.dissimilar_updates
        + diagnostics.similar_violations
        == diagnostics.sampled_pairs
    )
    assert diagnostics.similar_positive_steps == diagnostics.similar_violations
    assert diagnostics.similar_zero_steps == 0
    assert 0 <= diagnostics.final_rank <= X.shape[1]
    assert diagnostics.minimum_eigenvalue >= -1e-10


def test_paper_fit_diagnostics_account_for_zero_steps() -> None:
    """Paper-mode similar violations are classified as positive or zero steps."""
    X, y = _dataset()
    model = RDML(max_iter=250, random_state=13, update_method="paper").fit(X, y)
    diagnostics = model.diagnostics_

    assert diagnostics.sampled_pairs == 250
    assert (
        diagnostics.correct_pairs
        + diagnostics.dissimilar_updates
        + diagnostics.similar_violations
        == diagnostics.sampled_pairs
    )
    assert (
        diagnostics.similar_positive_steps + diagnostics.similar_zero_steps
        == diagnostics.similar_violations
    )
    assert diagnostics.similar_zero_steps >= 0
    assert 0 <= diagnostics.final_rank <= X.shape[1]
    assert diagnostics.minimum_eigenvalue >= -1e-10


def test_fit_diagnostics_are_reproducible_with_fixed_seed() -> None:
    """Identical pair streams produce identical diagnostic summaries."""
    X, y = _dataset()
    first = RDML(max_iter=100, random_state=42, update_method="paper").fit(X, y)
    second = RDML(max_iter=100, random_state=42, update_method="paper").fit(X, y)

    assert first.diagnostics_ == second.diagnostics_
