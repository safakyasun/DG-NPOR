import numpy as np

from por_core.neural_geometry import NeuralGeometryMap
from por_hep.atlas_jetset import RECONSTRUCTED_TRACK_INPUT_FIELDS
from por_hep.self_configuring_geometry import (
    PhysicsConstrainedGatedTrackGeometry,
    dropout_stability_audit,
)


class ArrayTrackSource:
    def __init__(self, tracks, mask, jets):
        self.tracks = tracks
        self.mask = mask
        self.jets = jets

    def read_many(self, indices, batch_size=2048):
        indices = np.asarray(indices, dtype=int)
        return self.tracks[indices], self.mask[indices], self.jets[indices]

    def read(self, indices):
        return self.read_many(indices)


def synthetic_tracks(seed=17, rows=240, tracks_per_jet=7):
    rng = np.random.RandomState(seed)
    labels = np.repeat([0, 1], rows // 2)
    rng.shuffle(labels)
    fields = len(RECONSTRUCTED_TRACK_INPUT_FIELDS)
    tracks = rng.normal(0.0, 0.7, size=(rows, tracks_per_jet, fields)).astype(
        np.float32
    )
    mask = rng.uniform(size=(rows, tracks_per_jet)) > 0.12
    mask[:, 0] = True
    d0 = RECONSTRUCTED_TRACK_INPUT_FIELDS.index(
        "lifetimeSignedD0Significance"
    )
    z0 = RECONSTRUCTED_TRACK_INPUT_FIELDS.index(
        "lifetimeSignedZ0SinThetaSignificance"
    )
    tracks[labels == 1, 0, d0] += 4.5
    tracks[labels == 1, 0, z0] += 3.5
    tracks *= mask[:, :, None]
    jets = rng.normal(size=(rows, 4)).astype(np.float32)
    jets[:, 3] = mask.sum(axis=1)
    return tracks, mask, jets, labels


def test_sparse_gated_encoder_fits_and_hardens_legal_branches():
    tracks, mask, jets, labels = synthetic_tracks()
    source = ArrayTrackSource(tracks, mask, jets)
    model = PhysicsConstrainedGatedTrackGeometry(
        hidden_dimension=10,
        geometry_dimension=12,
        geometry_depth=2,
        attention_heads=2,
        max_tracks=7,
        graph_neighbors=3,
        sequential_steps=2,
        gate_l0=5e-4,
        batch_size=48,
        max_epochs=5,
        patience=3,
        learning_rate=4e-3,
        random_state=23,
    ).fit(
        source,
        np.arange(0, 170),
        labels[:170],
        np.arange(170, 240),
        labels[170:],
    )
    result = model.encode_source(source, np.arange(170, 240), batch_size=31)
    summary = model.summary()
    assert result["geometry"].shape == (70, 12)
    assert np.isfinite(result["geometry"]).all()
    assert summary["hard_branch_gates"]["H"] == 1
    assert len(summary["selected_branches"]) >= 2
    assert summary["attention_heads"] == 2
    assert summary["geometry_depth"] == 2
    assert summary["auxiliary_probability_used_in_final_prediction"] is False


def test_finalized_encoder_is_permutation_invariant():
    tracks, mask, jets, labels = synthetic_tracks(rows=160)
    source = ArrayTrackSource(tracks, mask, jets)
    model = PhysicsConstrainedGatedTrackGeometry(
        hidden_dimension=8,
        geometry_dimension=10,
        geometry_depth=1,
        attention_heads=2,
        max_tracks=7,
        graph_neighbors=3,
        sequential_steps=2,
        batch_size=40,
        max_epochs=3,
        patience=2,
        random_state=31,
    ).fit(source, np.arange(110), labels[:110], np.arange(110, 160), labels[110:])
    permutation = np.array([4, 1, 6, 0, 3, 5, 2])
    original, _ = model.transform_arrays(tracks[:20], mask[:20], jets[:20])
    permuted, _ = model.transform_arrays(
        tracks[:20, permutation], mask[:20, permutation], jets[:20]
    )
    np.testing.assert_allclose(original, permuted, atol=3e-5, rtol=3e-5)


def test_dropout_stability_audit_is_finite_and_nonnegative():
    tracks, mask, jets, labels = synthetic_tracks(rows=160)
    source = ArrayTrackSource(tracks, mask, jets)
    model = PhysicsConstrainedGatedTrackGeometry(
        hidden_dimension=8,
        geometry_dimension=10,
        max_tracks=7,
        sequential_steps=2,
        batch_size=40,
        max_epochs=2,
        patience=2,
        random_state=37,
    ).fit(source, np.arange(110), labels[:110], np.arange(110, 160), labels[110:])
    result = dropout_stability_audit(
        model, source, np.arange(110, 160), random_state=41
    )
    assert np.isfinite(result["mean_absolute_probability_shift"])
    assert result["mean_absolute_probability_shift"] >= 0.0


def test_h_theta_architecture_families_preserve_input_dimension():
    rng = np.random.RandomState(43)
    X = rng.normal(size=(180, 8))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    for architecture in ("shallow", "standard", "deep", "tapered"):
        model = NeuralGeometryMap(
            max_epochs=3,
            patience=2,
            batch_size=64,
            architecture=architecture,
            random_state=47,
        ).fit(X, y)
        transformed = model.transform(X[:12])
        assert transformed.shape == (12, 8)
        assert model.summary()["architecture"] == architecture
