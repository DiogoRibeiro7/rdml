"""Compare Euclidean and RDML k-NN on a reproducible synthetic problem."""

from __future__ import annotations

import numpy as np

from rdml import RDML, accuracy_score, knn_predict


def make_dataset(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Create a binary problem with informative and nuisance dimensions."""
    rng = np.random.default_rng(seed)
    samples_per_class = 120
    informative_scale = 0.7
    nuisance_scale = 3.0

    class_zero_signal = rng.normal(
        loc=(-1.0, 0.0), scale=informative_scale, size=(samples_per_class, 2)
    )
    class_one_signal = rng.normal(
        loc=(1.0, 0.0), scale=informative_scale, size=(samples_per_class, 2)
    )
    nuisance_zero = rng.normal(scale=nuisance_scale, size=(samples_per_class, 8))
    nuisance_one = rng.normal(scale=nuisance_scale, size=(samples_per_class, 8))

    X = np.vstack(
        [
            np.hstack([class_zero_signal, nuisance_zero]),
            np.hstack([class_one_signal, nuisance_one]),
        ]
    )
    y = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def main() -> None:
    """Fit RDML once and report raw-space and learned-space k-NN accuracy."""
    X, y = make_dataset()
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    euclidean_predictions = knn_predict(X_train, y_train, X_test, n_neighbors=5)
    euclidean_accuracy = accuracy_score(y_test, euclidean_predictions)

    model = RDML(
        learning_rate=0.02,
        max_iter=5_000,
        margin=1.0,
        random_state=2026,
        update_method="paper",
    ).fit(X_train, y_train)
    X_train_metric = model.transform(X_train)
    X_test_metric = model.transform(X_test)
    rdml_predictions = knn_predict(
        X_train_metric,
        y_train,
        X_test_metric,
        n_neighbors=5,
    )
    rdml_accuracy = accuracy_score(y_test, rdml_predictions)

    print(f"Euclidean 5-NN accuracy: {euclidean_accuracy:.3f}")
    print(f"RDML 5-NN accuracy:      {rdml_accuracy:.3f}")
    print(f"Difference:               {rdml_accuracy - euclidean_accuracy:+.3f}")


if __name__ == "__main__":
    main()
