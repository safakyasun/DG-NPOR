import numpy as np

from por_core.readout import StructuredOrbitalMeasurement


def test_structured_measurement_analytic_gradient_and_convergence():
    rng = np.random.RandomState(123)
    state = rng.normal(size=(140, 6))
    state /= np.linalg.norm(state, axis=1, keepdims=True)
    labels = (state[:, 0] + 0.6 * state[:, 1] * state[:, 2] > 0).astype(int)
    measurement = StructuredOrbitalMeasurement(
        rank=3, max_iter=1200, n_restarts=1, random_state=5
    ).fit(state, labels)
    assert measurement.optimization_result_.success
    theta = measurement.B_.reshape(-1)
    _, analytic = measurement._objective_and_gradient_(theta)
    step = 1e-6
    for index in np.linspace(0, len(theta) - 1, 12, dtype=int):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference = (
            measurement._objective_and_gradient_(plus)[0]
            - measurement._objective_and_gradient_(minus)[0]
        ) / (2.0 * step)
        relative_error = abs(finite_difference - analytic[index]) / max(
            1.0, abs(finite_difference), abs(analytic[index])
        )
        assert relative_error < 2e-5
    np.testing.assert_allclose(
        measurement.predict_proba(state).sum(axis=1), 1.0, atol=1e-10
    )
