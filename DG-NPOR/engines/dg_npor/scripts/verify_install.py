#!/usr/bin/env python3
"""Small deterministic derivative-gated Neural-POR verification."""

import numpy as np
from sklearn.model_selection import train_test_split

from por_core import DerivativeGatedNeuralPOR


def main():
    rng = np.random.RandomState(17)
    X = rng.normal(size=(720, 10))
    score = 2.0 * X[:, 0] - 1.4 * X[:, 1] + 0.6 * X[:, 2]
    y = (score + 0.35 * rng.normal(size=len(X)) > 0.0).astype(int)
    development, test = train_test_split(
        np.arange(len(X)), test_size=120, stratify=y, random_state=1
    )
    base, remain = train_test_split(
        development, train_size=320, stratify=y[development], random_state=2
    )
    operator, audit = train_test_split(
        remain, train_size=120, stratify=y[remain], random_state=3
    )
    model = DerivativeGatedNeuralPOR(
        n_neighbors=12,
        scale_neighbor=6,
        lambda_pi_grid=(0.25, 0.5, 1.0),
        k_max=24,
        dimension_bootstrap_repetitions=10,
        neural_max_epochs=45,
        neural_patience=8,
        neural_batch_size=64,
        measurement_rank=3,
        measurement_max_iter=300,
        measurement_restarts=1,
        max_graph_samples=360,
        max_measurement_samples=180,
        random_state=9,
    ).fit(
        X[base], y[base],
        X_operator_validation=X[operator],
        y_operator_validation=y[operator],
        X_dimension_validation=X[audit],
        y_dimension_validation=y[audit],
        X_reference_train=X[base],
        X_reference_dimension=X[audit],
    )
    probability = model.predict_proba(X[test])
    assert probability.shape == (len(test), 2)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-8)
    assert model.selected_k_ >= 2
    assert len(model.selected_orbital_indices_) == model.selected_k_
    assert model.criterion_reached_
    assert not model.selected_[
        "neural_auxiliary_probability_used_for_final_prediction"
    ]
    assert np.max(np.abs(
        np.sum(model.measurement_.M_, axis=0) - np.eye(model.selected_k_)
    )) < 1e-6
    print(
        "Verification passed: K_DG=%d, orbitals=%s, f(x)=%s, "
        "loss-derivative=false, neural-aux-fusion=false"
        % (
            model.selected_k_,
            [int(value + 1) for value in model.selected_orbital_indices_],
            model.transform(X[test[:12]]).shape,
        )
    )


if __name__ == "__main__":
    main()
