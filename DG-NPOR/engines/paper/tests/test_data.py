from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from jetset_litcomp.data import (
    LABEL_FIELD,
    TRACK_FIELDS,
    JetSetBatchReader,
    RobustTrackScaler,
)


def _fixture(path: Path):
    jet_dtype = np.dtype(
        [(LABEL_FIELD, "i4"), ("eventNumber", "i8"), ("pt_btagJes", "f4"), ("eta_btagJes", "f4")]
    )
    track_dtype = np.dtype([("valid", "?")] + [(name, "f4") for name in TRACK_FIELDS])
    jets = np.zeros(4, dtype=jet_dtype)
    jets[LABEL_FIELD] = [0, 5, 0, 5]
    jets["eventNumber"] = [10, 11, 12, 13]
    jets["pt_btagJes"] = [30_000, 40_000, 50_000, 60_000]
    jets["eta_btagJes"] = [-1.0, 0.2, 1.2, -0.5]
    tracks = np.zeros((4, 40), dtype=track_dtype)
    tracks["valid"][:, :3] = True
    for index, name in enumerate(TRACK_FIELDS):
        tracks[name][:, :3] = np.arange(12, dtype=float).reshape(4, 3) + index
    tracks["lifetimeSignedD0Significance"][:, :3] = [1.0, 7.0, 3.0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("jets", data=jets, chunks=(2,))
        handle.create_dataset("tracks", data=tracks, chunks=(2, 40))


def test_reader_alignment_and_training_only_scaler(tmp_path):
    path = tmp_path / "tiny.h5"
    _fixture(path)
    with JetSetBatchReader(path, np.arange(4), max_tracks=2) as reader:
        labels, events = reader.read_labels_and_events()
        tracks, mask, context = reader.read(np.array([3, 1]))
    assert labels.tolist() == [0, 1, 0, 1]
    assert events.tolist() == [10, 11, 12, 13]
    assert tracks.shape == (2, 2, len(TRACK_FIELDS))
    assert mask.all()
    assert np.allclose(context[:, 3], 3)
    scaler = RobustTrackScaler().fit(tracks, mask, context)
    transformed, transformed_mask, transformed_context = scaler.transform(
        tracks, mask, context
    )
    assert transformed.shape == tracks.shape
    assert transformed_mask.shape == mask.shape
    assert transformed_context.shape == context.shape
    assert np.isfinite(transformed).all()
