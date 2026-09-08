"""Compare exact PSD projection with the paper's adaptive rank-one update."""

from __future__ import annotations

import time

import numpy as np

from rdml import paper_step_size, project_psd


def _spd_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a well-conditioned positive-definite matrix."""
    basis = rng.normal(size=(dimension, dimension))
    return basis @ basis.T + np.eye(dimension)


def benchmark(dimension: int, *, repeats: int = 20, seed: int = 42) -> None:
    """Print runtime and numerical difference for one matrix dimension."""
    rng = np.random.default_rng(seed)
    metric = _spd_matrix(dimension, rng)
    difference = rng.normal(size=dimension)
    learning_rate = 0.01
    candidate = metric - learning_rate * np.outer(difference, difference)

    start = time.perf_counter()
    for _ in range(repeats):
        exact = project_psd(candidate)
    exact_seconds = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        step = paper_step_size(metric, difference, 1.0, learning_rate)
        paper = metric - step * np.outer(difference, difference)
    paper_seconds = (time.perf_counter() - start) / repeats

    frobenius_error = float(np.linalg.norm(exact - paper, ord="fro"))
    minimum_eigenvalue = float(np.linalg.eigvalsh(paper).min())

    print(
        f"d={dimension:4d}  exact={exact_seconds:.6f}s  "
        f"paper={paper_seconds:.6f}s  speedup={exact_seconds / paper_seconds:.2f}x  "
        f"fro_error={frobenius_error:.3e}  min_eig={minimum_eigenvalue:.3e}"
    )


if __name__ == "__main__":
    for size in (16, 32, 64, 128, 256):
        benchmark(size)
