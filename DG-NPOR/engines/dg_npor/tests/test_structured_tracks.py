import numpy as np
import h5py

from por_hep.atlas_jetset import (
    RECONSTRUCTED_TRACK_INPUT_FIELDS,
    _required_schema,
)
from por_hep.structured_tracks import (
    H5StructuredTrackSource,
    IdentityFeatureMap,
    STRUCTURED_REPRESENTATIONS,
    StructuredTrackGeometry,
    StructuredTrackScaler,
    load_atlas_jetset_index,
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


def synthetic_tracks(seed=7, rows=360, tracks_per_jet=8):
    rng = np.random.RandomState(seed)
    fields = len(RECONSTRUCTED_TRACK_INPUT_FIELDS)
    labels = np.repeat([0, 1], rows // 2)
    rng.shuffle(labels)
    tracks = rng.normal(0.0, 0.7, size=(rows, tracks_per_jet, fields)).astype(
        np.float32
    )
    mask = rng.uniform(size=(rows, tracks_per_jet)) > 0.12
    mask[:, 0] = True
    d0_sig = RECONSTRUCTED_TRACK_INPUT_FIELDS.index(
        "lifetimeSignedD0Significance"
    )
    z0_sig = RECONSTRUCTED_TRACK_INPUT_FIELDS.index(
        "lifetimeSignedZ0SinThetaSignificance"
    )
    tracks[labels == 1, 0, d0_sig] += 5.0
    tracks[labels == 1, 0, z0_sig] += 4.0
    tracks *= mask[:, :, None]
    jets = rng.normal(size=(rows, 4)).astype(np.float32)
    jets[:, 3] = mask.sum(axis=1)
    return tracks, mask, jets, labels


def test_scaler_preserves_mask_and_finiteness():
    tracks, mask, jets, _ = synthetic_tracks(rows=40)
    scaler = StructuredTrackScaler().fit(tracks, mask, jets)
    transformed, transformed_mask, jet_values = scaler.transform(
        tracks, mask, jets
    )
    assert transformed.shape == tracks.shape
    assert np.array_equal(transformed_mask, mask)
    assert np.all(transformed[~mask] == 0.0)
    assert np.isfinite(transformed).all()
    assert np.isfinite(jet_values).all()


def test_set_encoder_learns_geometry_without_using_auxiliary_as_output():
    tracks, mask, jets, labels = synthetic_tracks()
    source = ArrayTrackSource(tracks, mask, jets)
    train = np.arange(0, 280)
    validation = np.arange(280, 360)
    model = StructuredTrackGeometry(
        representation="set",
        hidden_dimension=10,
        geometry_dimension=12,
        max_tracks=8,
        batch_size=64,
        max_epochs=10,
        patience=4,
        learning_rate=4e-3,
        random_state=11,
    ).fit(
        source,
        train,
        labels[train],
        validation,
        labels[validation],
    )
    result = model.encode_source(source, validation, batch_size=31)
    assert result["geometry"].shape == (len(validation), 12)
    assert result["auxiliary_probability"].shape == (len(validation),)
    assert np.isfinite(result["geometry"]).all()
    assert model.summary()["auxiliary_probability_used_in_final_prediction"] is False
    assert model.validation_auc_ > 0.75


def test_all_relational_geometries_are_permutation_invariant():
    tracks, mask, jets, labels = synthetic_tracks(rows=120)
    permutation = np.array([5, 1, 7, 0, 3, 6, 2, 4])
    for representation in ("graph", "sequential", "parallel", "hybrid"):
        model = StructuredTrackGeometry(
            representation=representation,
            hidden_dimension=8,
            geometry_dimension=10,
            max_tracks=8,
            graph_neighbors=3,
            sequential_steps=3,
            random_state=17,
        )
        model.scaler_ = StructuredTrackScaler().fit(tracks[:80], mask[:80], jets[:80])
        scaled = model.scaler_.transform(tracks[:1], mask[:1], jets[:1])
        model._initialize(model._node_input(scaled[0], scaled[1]).shape[2])
        model.embedding_center_ = np.zeros(10, dtype=np.float32)
        model.embedding_scale_ = np.ones(10, dtype=np.float32)
        original, _ = model.transform_arrays(tracks[:12], mask[:12], jets[:12])
        permuted, _ = model.transform_arrays(
            tracks[:12, permutation], mask[:12, permutation], jets[:12]
        )
        np.testing.assert_allclose(original, permuted, atol=2e-5, rtol=2e-5)


def test_multiscale_channel_dimensions_and_dense_parallel_broadcast():
    tracks, mask, jets, _ = synthetic_tracks(rows=12)
    scaler = StructuredTrackScaler().fit(tracks, mask, jets)
    scaled_tracks, scaled_mask, _ = scaler.transform(tracks, mask, jets)
    expected_multipliers = {
        "graph": 4,
        "sequential": 4,
        "parallel": 4,
        "hybrid": 6,
    }
    assert set(expected_multipliers).issubset(STRUCTURED_REPRESENTATIONS)
    for representation, multiplier in expected_multipliers.items():
        model = StructuredTrackGeometry(
            representation=representation,
            hidden_dimension=8,
            geometry_dimension=10,
            max_tracks=8,
            graph_neighbors=3,
            sequential_steps=3,
        )
        node_input = model._node_input(scaled_tracks, scaled_mask)
        assert node_input.shape[2] == multiplier * len(
            RECONSTRUCTED_TRACK_INPUT_FIELDS
        )
    parallel = StructuredTrackGeometry(
        representation="parallel", max_tracks=8, parallel_floor=0.02
    )
    adjacency = parallel._parallel_adjacency(scaled_tracks, scaled_mask)
    valid_rows = scaled_mask & (scaled_mask.sum(axis=1, keepdims=True) > 1)
    np.testing.assert_allclose(adjacency.sum(axis=2)[valid_rows], 1.0, atol=1e-6)
    pair_valid = scaled_mask[:, :, None] & scaled_mask[:, None, :]
    diagonal = np.eye(mask.shape[1], dtype=bool)[None, :, :]
    assert np.all(adjacency[pair_valid & ~diagonal] > 0.0)


def test_hybrid_summary_declares_sequential_and_parallel_channels():
    tracks, mask, jets, labels = synthetic_tracks(rows=120)
    source = ArrayTrackSource(tracks, mask, jets)
    model = StructuredTrackGeometry(
        representation="hybrid",
        hidden_dimension=8,
        geometry_dimension=10,
        max_tracks=8,
        graph_neighbors=3,
        sequential_steps=3,
        batch_size=32,
        max_epochs=2,
        patience=2,
        random_state=29,
    ).fit(source, np.arange(80), labels[:80], np.arange(80, 120), labels[80:])
    summary = model.summary()
    assert summary["sequential_local_steps"] == 3
    assert summary["dense_parallel_broadcast"] is True
    assert summary["original_track_residual_channel_preserved"] is True


def test_identity_feature_map_rejects_dimension_change():
    adapter = IdentityFeatureMap(5)
    values = np.ones((4, 5))
    np.testing.assert_array_equal(adapter.fit_transform(values), values)
    try:
        adapter.transform(np.ones((4, 4)))
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("IdentityFeatureMap accepted the wrong dimension.")


def test_index_loader_and_h5_track_reader_leave_tracks_streamed(tmp_path):
    path = tmp_path / "synthetic.h5"
    required = _required_schema("HadronConeExclTruthLabelID")
    jet_dtype = []
    for name in sorted(required["jets"]):
        dtype = np.int64 if name == "eventNumber" else np.float32
        if name == "HadronConeExclTruthLabelID":
            dtype = np.int32
        jet_dtype.append((name, dtype))
    track_dtype = [
        (name, np.float32) for name in sorted(required["tracks"] - {"valid"})
    ] + [("valid", np.bool_)]
    jets = np.zeros(20, dtype=np.dtype(jet_dtype))
    jets["eventNumber"] = np.arange(20)
    jets["pt_btagJes"] = 50000.0 + np.arange(20)
    jets["eta_btagJes"] = 0.2
    jets["HadronConeExclTruthLabelID"][:10] = 0
    jets["HadronConeExclTruthLabelID"][10:] = 5
    for name in (
        "GN2v01_pb",
        "GN2v01_pc",
        "GN2v01_pu",
        "GN2v01_ptau",
        "DL1dv01_pb",
        "DL1dv01_pc",
        "DL1dv01_pu",
    ):
        jets[name] = 0.25
    tracks = np.zeros((20, 40), dtype=np.dtype(track_dtype))
    tracks["valid"][:, :3] = True
    tracks["lifetimeSignedD0Significance"][:, 0] = np.arange(20) + 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("jets", data=jets)
        handle.create_dataset("tracks", data=tracks)
        handle.create_dataset("eventwise", data=np.zeros(20, dtype=np.float32))
        handle.create_dataset("truth_hadrons", data=np.zeros(1, dtype=np.float32))
    index = load_atlas_jetset_index(
        path,
        sample=20,
        protocol="pilot-b-light",
        max_source_events=0,
        random_state=3,
    )
    assert len(index.y) == 20
    assert index.schema_audit["structured_tracks_materialized_at_index_stage"] is False
    with H5StructuredTrackSource(path, index.row_index, max_tracks=3) as source:
        aligned, mask, context = source.read(np.array([15, 1, 8]))
    assert aligned.shape == (3, 3, len(RECONSTRUCTED_TRACK_INPUT_FIELDS))
    assert mask.shape == (3, 3)
    np.testing.assert_allclose(
        np.expm1(context[:, 0]) * 1000.0,
        jets["pt_btagJes"][[15, 1, 8]],
        rtol=1e-5,
    )
