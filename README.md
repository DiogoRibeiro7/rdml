# Regularized Distance Metric Learning

A modern research-software implementation of the online **Regularized Distance Metric Learning (RDML)** algorithm introduced by Rong Jin, Shijun Wang, and Yang Zhou at NeurIPS 2009.

The repository is being modernized from its original research-prototype form into a typed, tested, reproducible Python package. The new `rdml` package is the canonical implementation. Historical root-level scripts are retained temporarily while their additional algorithms are reviewed and separated into coherent modules.

## Mathematical reference

For labelled samples \((x_i, y_i)\), the paper learns a positive semi-definite matrix \(A\) defining

\[
d_A^2(x_i, x_j) = (x_i-x_j)^\top A(x_i-x_j).
\]

For a sampled pair, define \(y_{ij}=+1\) when the labels agree and \(-1\) otherwise. With margin \(b\), Algorithm 1 considers the pair correctly classified when

\[
y_{ij}\left(b-d_A^2(x_i,x_j)\right) > 0.
\]

A violating pair receives the online update

\[
A_t = \Pi_{S_+}\left[A_{t-1}-\lambda y_{ij}(x_i-x_j)(x_i-x_j)^\top\right],
\]

where \(\Pi_{S_+}\) denotes projection onto the positive semi-definite cone.

The current package implements this **exact projected update**. This is deliberately used as a correctness reference before implementing the faster approximate projection derived later in the paper.

## Installation

The project uses Poetry and supports Python 3.11–3.13.

```bash
poetry install --with dev
```

## Usage

```python
import numpy as np

from rdml import RDML

X = np.array(
    [
        [0.0, 0.0],
        [0.2, 0.1],
        [2.0, 0.0],
        [2.2, 0.1],
    ]
)
y = np.array([0, 0, 1, 1])

model = RDML(
    learning_rate=0.1,
    max_iter=1_000,
    margin=1.0,
    random_state=42,
).fit(X, y)

A = model.metric_
X_metric = model.transform(X)
```

`model.metric_` is the learned PSD matrix. `transform` returns coordinates in a Euclidean space whose squared Euclidean distances reproduce the learned Mahalanobis distances.

## Development

The engineering baseline is intentionally strict for the canonical package and its tests:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src
poetry run pytest
```

CI runs these checks on Python 3.11, 3.12, and 3.13. Tests enforce at least 90% branch-aware coverage of the modern `rdml` package. The two historical root-level scripts are intentionally excluded from Ruff and pre-commit until their algorithms are reviewed and extracted; they remain preserved as legacy reference code rather than being silently reformatted or behaviourally changed.

## Modernization roadmap

1. Establish the exact projected RDML implementation as a tested mathematical reference.
2. Implement the paper's efficient approximate PSD-preserving update and test it against the exact baseline.
3. Add k-NN evaluation helpers and reproduce selected experiments from the paper on redistributable datasets.
4. Separate and validate the historical low-rank bilinear and OASIS-style implementations.
5. Add research documentation covering derivations, assumptions, complexity, and reproducibility.

## Reference

Rong Jin, Shijun Wang, and Yang Zhou. **Regularized Distance Metric Learning: Theory and Algorithm.** Advances in Neural Information Processing Systems 22, 2009, pp. 862–870.

Paper: https://proceedings.neurips.cc/paper/2009/hash/a666587afda6e89aec274a3657558a27-Abstract.html

## License

GNU General Public License v3.0 or later. See `LICENSE`.
