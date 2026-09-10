import numpy as np

from por_core.predictive_field_dimension import (
    select_predictive_field_dimension,
)


def _posterior_from_field(field):
    probability = 1.0 / (1.0 + np.exp(-field))
    return np.column_stack([1.0 - probability, probability])


def test_unpenalized_predictive_field_support_is_compact_and_stable():
    rng = np.random.RandomState(4)
    Q, _ = np.linalg.qr(rng.normal(size=(300, 8)))
    field = 4.0 * Q[:, 0] + 2.0 * Q[:, 1]
    result = select_predictive_field_dimension(
        phi=Q,
        eigenvalues=np.arange(8, dtype=float),
        degree=np.ones(300),
        posterior=_posterior_from_field(field),
        priors=np.array([0.5, 0.5]),
        checkpoints=(4, 8),
        bootstrap_repetitions=20,
        random_state=5,
    )
    assert result.selected_k == 2
    assert result.spectrum_stable
    assert 1.0 <= result.participation_rank <= 2.0
    assert result.captured_weight > 0.99
    assert result.summary()["extra_eigenvalue_penalty"] is False


def test_degenerate_block_rotation_does_not_change_dimension_or_weights():
    rng = np.random.RandomState(8)
    Q, _ = np.linalg.qr(rng.normal(size=(240, 6)))
    field = 3.0 * Q[:, 0] - 2.0 * Q[:, 1] + 0.2 * Q[:, 2]
    common = dict(
        eigenvalues=np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        degree=np.ones(240),
        posterior=_posterior_from_field(field),
        priors=np.array([0.5, 0.5]),
        checkpoints=(3, 6),
        bootstrap_repetitions=0,
    )
    first = select_predictive_field_dimension(phi=Q, **common)
    angle = 0.73
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated = Q.copy()
    rotated[:, :2] = Q[:, :2] @ rotation
    second = select_predictive_field_dimension(phi=rotated, **common)
    assert first.selected_k == second.selected_k
    np.testing.assert_allclose(
        first.normalized_weights, second.normalized_weights, atol=1e-10
    )
