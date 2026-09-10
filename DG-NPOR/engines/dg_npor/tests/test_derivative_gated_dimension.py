import numpy as np

from por_core.derivative_gated_dimension import (
    class_occupancy_screening_measurement,
    predictive_information_gate_gradient,
    select_derivative_gated_dimension,
)


def _information(probability, priors):
    p = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return (
        p * np.log(p / priors[1])
        + (1.0 - p) * np.log((1.0 - p) / priors[0])
    )


def test_analytic_gate_derivative_matches_finite_difference_of_J():
    rng = np.random.RandomState(2)
    A = rng.normal(size=(20, 6))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    y = np.asarray([0, 1] * 10)
    priors = np.array([0.5, 0.5])
    M = class_occupancy_screening_measurement(A, y, rho=1e-2)
    analytic, _ = predictive_information_gate_gradient(A, M[1], priors)
    step = 1e-6
    numerical = np.empty_like(analytic)
    for orbital in range(A.shape[1]):
        gates_plus = np.ones(A.shape[1])
        gates_minus = np.ones(A.shape[1])
        gates_plus[orbital] += step
        gates_minus[orbital] -= step
        plus = A * gates_plus
        minus = A * gates_minus
        plus /= np.linalg.norm(plus, axis=1, keepdims=True)
        minus /= np.linalg.norm(minus, axis=1, keepdims=True)
        p_plus = np.einsum("ni,ij,nj->n", plus, M[1], plus)
        p_minus = np.einsum("ni,ij,nj->n", minus, M[1], minus)
        numerical[:, orbital] = (
            _information(p_plus, priors) - _information(p_minus, priors)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-4, atol=2e-6)


def test_sparse_selection_can_keep_high_index_without_intermediate_orbitals():
    rng = np.random.RandomState(7)
    Q_train, _ = np.linalg.qr(rng.normal(size=(500, 12)))
    Q_validation, _ = np.linalg.qr(rng.normal(size=(300, 12)))
    # Define class geometry through the high-index coordinates as well as the
    # first modes.  The selected set is allowed to be non-contiguous.
    train_score = (
        3.0 * Q_train[:, 0]
        - 2.5 * Q_train[:, 1]
        + 4.0 * Q_train[:, 9]
    )
    y = (train_score > np.median(train_score)).astype(int)
    # Align validation coordinates with the same class-oriented axes without
    # using validation labels in selection.
    Q_validation[:, 0] *= 3.0
    Q_validation[:, 1] *= 2.5
    Q_validation[:, 9] *= 4.0
    result = select_derivative_gated_dimension(
        phi_train=Q_train,
        phi_validation=Q_validation,
        eigenvalues=np.arange(12, dtype=float),
        y_train=y,
        priors=np.array([0.5, 0.5]),
        checkpoints=(6, 12),
        bootstrap_repetitions=20,
        random_state=11,
    )
    assert result.selected_k < int(np.max(result.selected_indices) + 1)
    assert result.summary()["classification_loss_derivative_used"] is False


def test_degenerate_block_selection_is_rotation_invariant():
    rng = np.random.RandomState(13)
    train, _ = np.linalg.qr(rng.normal(size=(420, 8)))
    validation, _ = np.linalg.qr(rng.normal(size=(240, 8)))
    y = (2.0 * train[:, 0] - 1.5 * train[:, 1] + train[:, 5] > 0).astype(int)
    eigenvalues = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    common = dict(
        eigenvalues=eigenvalues,
        y_train=y,
        priors=np.bincount(y, minlength=2) / len(y),
        checkpoints=(4, 8),
        bootstrap_repetitions=0,
    )
    first = select_derivative_gated_dimension(
        phi_train=train, phi_validation=validation, **common
    )
    angle = 0.61
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    train_rotated = train.copy()
    validation_rotated = validation.copy()
    train_rotated[:, :2] = train[:, :2] @ rotation
    validation_rotated[:, :2] = validation[:, :2] @ rotation
    second = select_derivative_gated_dimension(
        phi_train=train_rotated,
        phi_validation=validation_rotated,
        **common
    )
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_allclose(
        first.orbital_weights[:2], second.orbital_weights[:2], atol=1e-10
    )
