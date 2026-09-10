import numpy as np

from por_core.neural_geometry import NeuralGeometryMap


def test_neural_geometry_preserves_declared_dimension_and_has_no_fused_output():
    rng = np.random.RandomState(13)
    X = rng.normal(size=(260, 7))
    y = (X[:, 0] - 0.7 * X[:, 1] > 0.0).astype(int)
    geometry = NeuralGeometryMap(
        max_epochs=35, patience=7, batch_size=64, random_state=3
    ).fit(X, y)
    Z = geometry.transform(X[:20])
    assert Z.shape == (20, 7)
    assert np.isfinite(Z).all()
    summary = geometry.summary()
    assert summary["geometry_is_claimed_compact_embedding"] is False
    assert summary["auxiliary_probability_used_in_final_prediction"] is False
