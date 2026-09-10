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

No parameter search is performed to force agreement with Table 1.

## Running the benchmark

Install the optional benchmark dependency group:

```bash
poetry install --with benchmark
```

Then run:

```bash
poetry run python benchmarks/reproduce_paper_subset.py
```

The output reports mean classification error and sample standard deviation for Euclidean, exact RDML, and paper-safe RDML. For the RDML methods it also reports mean final rank, and for paper-safe it reports the fraction of similar-pair violations that receive a zero adaptive step.

## Published reference values

For the two datasets currently covered, Table 1 reports:

| Dataset | Euclidean error | Online-reg error |
| --- | ---: | ---: |
| Iris | 4.0% ± 1.7% | 3.2% ± 1.3% |
| Wine | 31.9% ± 2.8% | 1.8% ± 1.1% |

The next methodological task is to investigate the efficient-update singular-start convention and the remaining undocumented tuning/preprocessing assumptions before expanding Table 1 coverage.
