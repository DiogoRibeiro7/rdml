# Partial reproduction of Experiment I

This repository includes a protocol-aligned partial reproduction of Experiment I from Jin, Wang, and Zhou (2009).

## What is reproduced

The published experiment evaluates learned metrics with a 3-nearest-neighbour classifier. For each UCI dataset, 50% of the samples are selected for training and the remaining 50% are used for testing. The reported classification error is averaged over 10 runs.

The current reproduction covers two of the datasets used in the paper:

- Iris: 150 samples, 4 features, 3 classes
- Wine: 178 samples, 13 features, 3 classes

The script uses the copies bundled with scikit-learn, so the benchmark does not download data at runtime.

## What is not claimed

This is not a bit-for-bit replication of Table 1. The paper does not publish enough implementation detail to reconstruct every stochastic and tuning choice exactly. In particular, the published random train/test seeds and sampled pair streams are not available, and the paper does not give a complete reproducible configuration for `online-reg` on each dataset.

For that reason, the published Table 1 values are printed as reference context and are not used as regression-test targets.

## Fixed implementation choices

The reproduction makes every additional choice explicit:

- split seeds: integers 0 through 9
- split rule: NumPy random permutation, first half training and second half testing
- classifier: deterministic 3-NN from `rdml.knn_predict`
- RDML update: `update_method="paper"`
- learning rate: 0.1
- margin: 1.0
- sampled pair iterations: 10,000
- model seed: same integer as the split seed
- feature preprocessing: none

The lack of feature standardization is intentional and visible because it materially affects Euclidean performance on Wine. It should not be interpreted as proof that this exactly matches undocumented preprocessing in the original Matlab experiments.

## Running the benchmark

Install the optional benchmark dependency group:

```bash
poetry install --with benchmark
```

Then run:

```bash
poetry run python benchmarks/reproduce_paper_subset.py
```

The output reports mean classification error and sample standard deviation for both Euclidean distance and the current RDML implementation, followed by the corresponding published values for context.

## Published reference values

For the two datasets currently covered, Table 1 reports:

| Dataset | Euclidean error | Online-reg error |
| --- | ---: | ---: |
| Iris | 4.0% ± 1.7% | 3.2% ± 1.3% |
| Wine | 31.9% ± 2.8% | 1.8% ± 1.1% |

The next reproduction step is to add more of the original UCI datasets while preserving dataset provenance and avoiding silent preprocessing changes.
