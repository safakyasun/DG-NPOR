"""Frozen-split and HDF5 access for the final JetSet comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRACK_FIELDS = (
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinThetaSignificance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
)
LABEL_FIELD = "HadronConeExclTruthLabelID"
LIGHT_LABEL = 0
BOTTOM_LABEL = 5
COORDINATE_INDICES = (TRACK_FIELDS.index("deta"), TRACK_FIELDS.index("dphi"))


@dataclass(frozen=True)
class FrozenSplit:
    source_rows: np.ndarray
    event_numbers: np.ndarray
    base: np.ndarray
    operator_validation: np.ndarray
    derivative_gate: np.ndarray
    wp_validation: np.ndarray
    independent_test: np.ndarray

    @property
    def particle_net_train(self):
        """All disjoint non-test training roles, without duplicating dimension rows."""
        values = np.concatenate(
            [self.base, self.operator_validation, self.derivative_gate]
        )
        if len(np.unique(values)) != len(values):
            raise RuntimeError("Frozen ParticleNet training roles overlap.")
        return values.astype(int)


def load_frozen_split(selfconfig_output_dir) -> FrozenSplit:
    path = Path(selfconfig_output_dir) / "frozen_split_indices.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frozen split not found: {path}")
    with np.load(path) as values:
        required = {
            "sampled_source_rows",
            "sampled_event_numbers",
            "base",
            "operator_validation",
            "derivative_gate",
            "wp_validation",
            "independent_test",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"Frozen split is missing arrays: {missing}")
        split = FrozenSplit(
            source_rows=np.asarray(values["sampled_source_rows"], dtype=np.int64),
            event_numbers=np.asarray(values["sampled_event_numbers"], dtype=np.int64),
            base=np.asarray(values["base"], dtype=int),
            operator_validation=np.asarray(values["operator_validation"], dtype=int),
            derivative_gate=np.asarray(values["derivative_gate"], dtype=int),
            wp_validation=np.asarray(values["wp_validation"], dtype=int),
            independent_test=np.asarray(values["independent_test"], dtype=int),
        )
    n_rows = len(split.source_rows)
    if len(split.event_numbers) != n_rows:
        raise ValueError("Frozen source-row and event arrays differ in length.")
    for name in (
        "base",
        "operator_validation",
        "derivative_gate",
        "wp_validation",
        "independent_test",
    ):
        index = getattr(split, name)
        if np.any(index < 0) or np.any(index >= n_rows):
            raise ValueError(f"Frozen role {name} contains an invalid logical index.")
    development = np.concatenate(
        [split.particle_net_train, split.wp_validation]
    )
    if np.intersect1d(development, split.independent_test).size:
        raise RuntimeError("Frozen ParticleNet development roles overlap locked test.")
    if np.intersect1d(split.particle_net_train, split.wp_validation).size:
        raise RuntimeError("Frozen ParticleNet train and validation roles overlap.")
    if split.source_rows.ndim != 1 or np.any(split.source_rows < 0) or np.any(np.diff(split.source_rows) <= 0):
        raise ValueError("Frozen source rows must be unique and strictly increasing.")
    role_names = ("base", "operator_validation", "derivative_gate", "wp_validation", "independent_test")
    indices = [getattr(split, name) for name in role_names]
    together = np.concatenate(indices)
    if len(together) != n_rows or not np.array_equal(np.sort(together), np.arange(n_rows)):
        raise ValueError("Frozen roles must partition every sampled row exactly once.")
    event_sets = [set(split.event_numbers[index].tolist()) for index in indices]
    for i in range(len(event_sets)):
        for j in range(i + 1, len(event_sets)):
            if event_sets[i] & event_sets[j]:
                raise ValueError(f"Event leakage between {role_names[i]} and {role_names[j]}.")
    return split


def load_existing_locked_predictions(selfconfig_output_dir):
    path = Path(selfconfig_output_dir) / "self_configuring_locked_test_predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            "The existing locked-test prediction file is missing. Run "
            "score_self_configuring_locked_test.py in the SelfConfig workspace first."
        )
    frame = pd.read_csv(path)
    required = {
        "source_row",
        "event_number",
        "y_true_light0_b1",
        "self_configuring_por_probability_b",
        "GN2v01_paper_Db_audit_only",
        "DL1dv01_paper_Db_audit_only",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Existing prediction file is missing columns: {missing}")
    if frame["source_row"].duplicated().any():
        raise ValueError("Existing locked-test prediction rows are not unique.")
    if set(frame["y_true_light0_b1"].astype(int).unique().tolist()) != {0, 1}:
        raise ValueError("Existing locked-test labels are not light=0 and b=1.")
    return frame


def read_structured_rows(dataset, rows, fields):
    """Read strictly increasing HDF5 rows once per compressed source chunk."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("HDF5 row indices must be one-dimensional.")
    if len(rows) and np.any(np.diff(rows) <= 0):
        raise ValueError("HDF5 source rows must be strictly increasing.")
    fields = tuple(fields)
    wrapper = dataset.fields(fields)
    empty = wrapper[:0]
    output = np.empty((len(rows),) + dataset.shape[1:], dtype=empty.dtype)
    if len(rows) == 0:
        return output
    chunk = int(dataset.chunks[0] if dataset.chunks else 4096)
    chunk_ids = rows // chunk
    for chunk_id in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids == chunk_id)
        start = int(chunk_id * chunk)
        stop = min(start + chunk, int(dataset.shape[0]))
        block = wrapper[start:stop]
        output[positions] = block[rows[positions] - start]
    return output


def _signed_log1p(values, scale=1.0):
    values = np.asarray(values, dtype=np.float32) * float(scale)
    return np.sign(values) * np.log1p(np.abs(values))


def transform_track_field(name, values):
    values = np.asarray(values, dtype=np.float32)
    if name in {
        "d0",
        "z0SinTheta",
        "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinThetaSignificance",
    }:
        return _signed_log1p(values)
    if name == "qOverP":
        return _signed_log1p(values, scale=1000.0)
    if name == "qOverPUncertainty":
        return np.log1p(np.maximum(values * 1000.0, 0.0))
    if name in {"phiUncertainty", "thetaUncertainty"}:
        return np.log1p(np.maximum(values, 0.0))
    return values


class JetSetBatchReader:
    """Read aligned top-|d0 significance| tracks for frozen logical rows."""

    def __init__(self, h5_file, source_rows, max_tracks=20):
        self.path = str(Path(h5_file))
        self.source_rows = np.asarray(source_rows, dtype=np.int64)
        self.max_tracks = int(max_tracks)
        if not 1 <= self.max_tracks <= 40:
            raise ValueError("max_tracks must lie in [1, 40].")
        self.handle = None

    def __enter__(self):
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "Reading the official JetSet HDF5 file requires h5py. "
                "Install requirements.txt in the active environment."
            ) from error
        self.handle = h5py.File(self.path, "r")
        for dataset in ("jets", "tracks"):
            if dataset not in self.handle:
                raise ValueError(f"Official JetSet file is missing {dataset}.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _open(self):
        if self.handle is None:
            raise RuntimeError("JetSetBatchReader must be used as a context manager.")

    def read_labels_and_events(self):
        """Read labels/event numbers for every selected row and verify alignment."""
        self._open()
        jets = read_structured_rows(
            self.handle["jets"], self.source_rows, (LABEL_FIELD, "eventNumber")
        )
        raw = np.asarray(jets[LABEL_FIELD], dtype=int)
        if not set(np.unique(raw).tolist()).issubset({LIGHT_LABEL, BOTTOM_LABEL}):
            raise ValueError("Frozen sample contains a non-light/non-b truth label.")
        return (raw == BOTTOM_LABEL).astype(int), np.asarray(
            jets["eventNumber"], dtype=np.int64
        )

    def read(self, logical_indices):
        self._open()
        logical = np.asarray(logical_indices, dtype=int)
        if logical.ndim != 1 or len(np.unique(logical)) != len(logical):
            raise ValueError("Batch logical indices must be a unique vector.")
        source = self.source_rows[logical]
        order = np.argsort(source)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        sorted_rows = source[order]
        tracks = read_structured_rows(
            self.handle["tracks"], sorted_rows, ("valid",) + TRACK_FIELDS
        )[inverse]
        jets = read_structured_rows(
            self.handle["jets"], sorted_rows, ("pt_btagJes", "eta_btagJes")
        )[inverse]

        valid_all = np.asarray(tracks["valid"], dtype=bool)
        ranking = np.abs(
            np.nan_to_num(
                np.asarray(tracks["lifetimeSignedD0Significance"], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        ranking = np.where(valid_all, ranking, -np.inf)
        ranked = np.argsort(-ranking, axis=1, kind="stable")[:, : self.max_tracks]
        mask = np.take_along_axis(valid_all, ranked, axis=1)
        columns = []
        for name in TRACK_FIELDS:
            values = transform_track_field(name, tracks[name])
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(np.take_along_axis(values, ranked, axis=1))
        aligned = np.stack(columns, axis=2).astype(np.float32)
        aligned *= mask[:, :, None]

        pt_gev = np.asarray(jets["pt_btagJes"], dtype=np.float32) / 1000.0
        eta = np.asarray(jets["eta_btagJes"], dtype=np.float32)
        context = np.column_stack(
            [
                np.log1p(np.maximum(pt_gev, 0.0)),
                eta,
                np.abs(eta),
                valid_all.sum(axis=1).astype(np.float32),
            ]
        ).astype(np.float32)
        return aligned, mask, context


class RobustTrackScaler:
    """Training-only median/(q90-q10) scaling for ParticleNet inputs."""

    def __init__(self, clip=10.0):
        self.clip = float(clip)

    @staticmethod
    def _fit_array(values):
        center = np.median(values, axis=0)
        scale = np.maximum(
            np.quantile(values, 0.90, axis=0)
            - np.quantile(values, 0.10, axis=0),
            1e-6,
        )
        return center.astype(np.float32), scale.astype(np.float32)

    def fit(self, tracks, mask, context):
        tracks = np.asarray(tracks, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        context = np.asarray(context, dtype=np.float32)
        if not np.any(mask):
            raise ValueError("Cannot fit a scaler without valid tracks.")
        self.track_center, self.track_scale = self._fit_array(tracks[mask])
        self.context_center, self.context_scale = self._fit_array(context)
        return self

    def transform(self, tracks, mask, context):
        if not hasattr(self, "track_center"):
            raise RuntimeError("RobustTrackScaler is not fitted.")
        mask = np.asarray(mask, dtype=bool)
        tracks = (np.asarray(tracks, dtype=np.float32) - self.track_center) / self.track_scale
        context = (
            np.asarray(context, dtype=np.float32) - self.context_center
        ) / self.context_scale
        tracks = np.clip(tracks, -self.clip, self.clip)
        context = np.clip(context, -self.clip, self.clip)
        tracks *= mask[:, :, None]
        return tracks.astype(np.float32), mask, context.astype(np.float32)

    def save(self, path):
        np.savez_compressed(
            path,
            track_center=self.track_center,
            track_scale=self.track_scale,
            context_center=self.context_center,
            context_scale=self.context_scale,
            clip=np.asarray([self.clip], dtype=float),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as values:
            scaler = cls(clip=float(values["clip"][0]))
            scaler.track_center = np.asarray(values["track_center"], dtype=np.float32)
            scaler.track_scale = np.asarray(values["track_scale"], dtype=np.float32)
            scaler.context_center = np.asarray(values["context_center"], dtype=np.float32)
            scaler.context_scale = np.asarray(values["context_scale"], dtype=np.float32)
        return scaler
