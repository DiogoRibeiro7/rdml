# Partial reproduction of Experiment I

This repository includes a protocol-aligned partial reproduction of Experiment I from Jin, Wang, and Zhou (2009).

## Reproduction provenance

![RDML partial reproduction provenance DAG](diagrams/rendered/reproduction_dag.svg)

The diagram separates what is inherited from the published experiment from what this repository must choose explicitly. It also shows why the published Table 1 values are **comparison context rather than optimisation or regression targets**: seeds, sampled pair streams, some tuning/preprocessing details, and the efficient-update singular-start convention are not fully specified by the paper.

The Graphviz source is [`diagrams/reproduction_dag.dot`](diagrams/reproduction_dag.dot). The committed SVG is regenerated in CI and checked for source/render drift.

## What is reproduced

The published experiment evaluates learned metrics with a 3-nearest-neighbour classifier. For each UCI dataset, 50% of the samples are selected for training and the remaining 50% are used for testing. The reported classification error is averaged over 10 runs.

The current reproduction covers two of the datasets used in the paper:

- Iris: 150 samples, 4 features, 3 classes
- Wine: 178 samples, 13 features, 3 classes

The script uses the copies bundled with scikit-learn, so the benchmark does not download data at runtime.

## What is not claimed

This is not a bit-for-bit replication of Table 1. The paper does not publish enough implementation detail to reconstruct every stochastic and tuning choice exactly. In particular, the published random train/test seeds and sampled pair streams are not available, and the paper does not give a complete reproducible configuration for `online-reg` on each dataset.

The paper also initializes the metric at zero while its efficient Theorem 6 step is written using an inverse of the current metric. The paper does not fully specify how the implementation resolves that singular-start case. This package therefore distinguishes the exact projected Algorithm 1 update from its conservative singular-safe interpretation of Theorem 6.

For these reasons, published Table 1 values are reference context rather than regression-test targets.

## Fixed implementation choices

The reproduction makes every additional choice explicit:

- split seeds: integers 0 through 9
- split rule: NumPy random permutation, first half training and second half testing
- classifier: deterministic 3-NN from `rdml.knn_predict`
- RDML learning rate: 0.1
- margin: 1.0
- sampled pair iterations: 10,000
- model seed: same integer as the split seed
- feature preprocessing: none

The lack of feature standardization is intentional and visible because it materially affects Euclidean performance on Wine. It should not be interpreted as proof that this exactly matches undocumented preprocessing in the original experiments.

## Diagnostic comparison

The reproduction reports three methods under the same splits and pair-stream seeds:

1. Euclidean 3-NN
2. `RDML(update_method="exact")`, the eigendecomposition-based projected Algorithm 1 reference
3. `RDML(update_method="paper")`, labelled **paper-safe** in the benchmark output because it combines Theorem 6 with this package's conservative singular-matrix rule

On the fixed seeds 0 through 9, the current implementation gives approximately:

| Dataset | Euclidean | RDML exact | RDML paper-safe | Paper online-reg |
| --- | ---: | ---: | ---: | ---: |
| Iris | 4.93% ± 2.36% | 4.80% ± 1.69% | 3.60% ± 2.09% | 3.2% ± 1.3% |
| Wine | 31.24% ± 4.06% | 12.70% ± 2.90% | 33.82% ± 3.72% | 1.8% ± 1.1% |

These numbers are diagnostics, not fitted targets. Iris is broadly compatible with the published pattern. Wine is not: the exact projected method improves strongly over Euclidean distance, while the current paper-safe path does not. That discrepancy is scientifically useful because it localizes the largest reproduction gap to the efficient-update interpretation rather than to the k-NN evaluator or the dataset itself.

### Structural fit diagnostics

`RDML.fit` exposes a frozen `FitDiagnostics` summary as `model.diagnostics_`. It records how the sampled pair stream is partitioned, how many similar-pair violations receive positive or zero adaptive steps, and the final numerical rank and minimum eigenvalue of the learned metric.

For the same ten Wine runs, the exact projected method finishes with mean numerical rank about **10.5 out of 13**. The paper-safe path finishes at mean rank about **1.3 out of 13**. Across the ten runs, all **34,148** similar-pair violations in the paper-safe path receive a zero adaptive step under the conservative singular-matrix rule.

Replacing conjugate gradient with an eigendecomposition-based exact evaluation of the same PSD-feasibility condition produces the same qualitative Wine behaviour. This shows that the observed stagnation is not primarily a CG convergence artifact: it follows from the singular feasible-update geometry itself under the zero initialization and raw Wine feature scales.

### Singular-start initialization sweep

`benchmarks/singular_start_sweep.py` changes only the initial metric used with the public Theorem 6 step. The stable estimator is not modified. Four named conventions are evaluated under identical splits and pair streams:

- `paper-zero`: \(A_0=0\), matching the current conservative paper-safe interpretation
- `paper-0.1I`: \(A_0=0.1I\)
- `paper-I`: \(A_0=I\)
- `paper-10I`: \(A_0=10I\)

The exact projected estimator remains the reference. On Wine, the fixed ten-run sweep gives approximately:

| Variant | Classification error | Mean final rank | Zero similar-pair steps |
| --- | ---: | ---: | ---: |
| exact | 12.70% ± 2.90% | 10.5 | 0% |
| `paper-zero` | 33.82% ± 3.72% | 1.3 | 100.00% |
| `paper-0.1I` | 30.79% ± 8.69% | 12.0 | 99.95% |
| `paper-I` | 18.31% ± 5.51% | 12.0 | 99.97% |
| `paper-10I` | 18.31% ± 5.51% | 12.0 | 99.97% |

A positive-definite start therefore changes the learned metric substantially, but it does **not** resolve the structural issue. Once a similar-pair subtraction reaches the PSD boundary, the metric becomes singular again and almost all subsequent similar-pair violations receive a zero step. The result also shows that the identity scale is a consequential modelling choice rather than a harmless numerical jitter.

This makes the next methodological question sharper: a faithful efficient variant needs an explicit convention for staying in, or returning to, the positive-definite interior rather than merely replacing the zero initialization with an arbitrary identity scale.

### Strict-interior reference sweep

`benchmarks/interior_step_sweep.py` tests a second question separately from
initialization: what happens if same-class updates are kept strictly inside the
positive-definite cone? Starting from \(A_0=I\), it uses the reference cap

\[
\alpha_t = \min\!\left(\lambda,\frac{\rho}{v^\top A_{t-1}^{-1}v}\right),
\qquad 0<\rho<1.
\]

A direct linear solve is used deliberately. This is a mathematical reference
for the convention, not yet a scalable implementation. The sweep compares
\(\rho\in\{0.01,0.05,0.1,0.2\}\) under the same ten Wine splits.

On raw Wine features, the interior cap improves substantially over the
boundary-hitting paper-safe path but remains sensitive to conditioning. The
best observed mean error in this fixed sweep is about **14.0%**, around
\(\rho=0.1\) to \(0.2\), versus **12.7%** for the exact projected reference.

Train-only standardization changes the picture dramatically: Euclidean 3-NN
itself drops from about **31.2%** error to about **5.1%**, while the small-
\(\rho\) interior variants are around **3.8–4.0%**. That is useful numerical
evidence, but it cannot be silently adopted as the paper protocol because the
published Wine Euclidean baseline is about **31.9%**. In other words,
standardization stabilizes the geometry while simultaneously changing the
benchmark problem.

This isolates two separate issues:

1. the boundary step can collapse the metric to a singular face of the PSD cone;
2. raw feature scaling can make inverse-based interior updates numerically
   fragile even when the algebraic update remains positive definite.

The next implementation question is therefore a stable factorized interior
update, evaluated without changing the published-data preprocessing by default.

### Factorized strict-interior reference

`benchmarks/factorized_interior.py` implements the same strict-interior
convention with a Cholesky factor \(A=L L^\top\) instead of repeated direct
matrix solves. For a same-class pair, it computes

\[
q=v^\top A^{-1}v = \|L^{-1}v\|^2,
\]

chooses \(\alpha=\min(\lambda,\rho/q)\), and applies a rank-one Cholesky
downdate. Different-class updates use the corresponding rank-one Cholesky
update. The per-step algebra is therefore \(O(d^2)\), with no explicit inverse
and no eigendecomposition.

At \(\rho=0.1\) and \(A_0=I\), the factorized path reproduces the direct
matrix-reference trajectory to numerical precision on fixed runs (relative
Frobenius discrepancy about \(2\times10^{-15}\) in a checked raw-Wine run).
Across the ten fixed Wine splits it gives the same mean classification error as
the direct reference: about **14.04%** on raw features and **4.04%** after
train-only standardization. Similar-pair zero steps disappear.

The conditioning warning remains visible, however. On raw Wine the smallest
Cholesky diagonal becomes extremely small, around \(10^{-75}\) on average in
the fixed sweep. The factorization therefore fixes the explicit-inverse and
singular-face mechanics, but it does not make badly scaled raw coordinates
well-conditioned.

This is still benchmark-only. The public estimator remains unchanged until the
factorized path has stronger numerical safeguards and tests.

No parameter search is performed to force agreement with Table 1.

## Running the benchmarks

Install the optional benchmark dependency group:

```bash
poetry install --with benchmark
```

Run the Experiment I subset:

```bash
poetry run python benchmarks/reproduce_paper_subset.py
```

Run the singular-start diagnostic sweep:

```bash
poetry run python benchmarks/singular_start_sweep.py
```

To run only the Wine diagnosis used in CI:

```bash
poetry run python benchmarks/singular_start_sweep.py --dataset wine
```

The main reproduction reports mean classification error and sample standard deviation for Euclidean, exact RDML, and paper-safe RDML. For the RDML methods it also reports mean final rank, and for paper-safe it reports the fraction of similar-pair violations that receive a zero adaptive step. The singular-start sweep reports the same structural quantities for each initialization convention.

## Published reference values

For the two datasets currently covered, Table 1 reports:

| Dataset | Euclidean error | Online-reg error |
| --- | ---: | ---: |
| Iris | 4.0% ± 1.7% | 3.2% ± 1.3% |
| Wine | 31.9% ± 2.8% | 1.8% ± 1.1% |

The next methodological task is to test explicitly named positive-definite-interior update conventions before expanding Table 1 coverage.
