"""Leakage-safe structured-track geometry for ATLAS JetSet DG-NPOR.

The original JetSet adapter independently summarized each reconstructed track
field. That is permutation invariant, but it removes the joint identity of a
track. This module instead keeps aligned per-track vectors, learns one shared
node encoder, and pools the encoded nodes into a jet geometry. Local,
sequential, dense-parallel, and hybrid variants use only reconstructed track
coordinates and impact-parameter information.

No truth-track field, GN2/DL1 score, GN2 vertex assignment, or auxiliary neural
probability is exposed to the final POR measurement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .atlas_jetset import (
    BOTTOM_LABEL,
    LABEL_FIELDS,
    LIGHT_LABEL,
    RECONSTRUCTED_TRACK_INPUT_FIELDS,
    _balanced_source_rows,
    _binary_audit_score,
    _protocol_eligible_mask,
    _read_rows_chunked,
    _track_transform,
    dl1d_paper_b_discriminant,
    gn2_paper_b_discriminant,
    inspect_schema,
)


STRUCTURED_REPRESENTATIONS = (
    "set",
    "graph",
    "sequential",
    "parallel",
    "hybrid",
)


@dataclass(frozen=True)
class AtlasJetSetIndex:
    """Selected JetSet rows without materializing the padded track tensor."""

    y: np.ndarray
    row_index: np.ndarray
    group_id: np.ndarray
    dataset_name: str
    audit_scores: dict
    audit_variables: dict
    label_field: str
    schema_audit: dict


def load_atlas_jetset_index(
    path,
    sample=20000,
    random_state=42,
    label_definition="cone",
    protocol="paper-ttbar",
    max_source_events=2000000,
):
    """Select the exact balanced sample while leaving tracks on disk."""
    path = Path(path)
    label_field = LABEL_FIELDS[str(label_definition)]
    schema = inspect_schema(path, label_definition=label_definition)
    with h5py.File(path, "r") as handle:
        labels_all = np.asarray(handle["jets"].fields(label_field)[:], dtype=int)
        pt_all = np.asarray(handle["jets"].fields("pt_btagJes")[:], dtype=float)
        eta_all = np.asarray(handle["jets"].fields("eta_btagJes")[:], dtype=float)
        events_all = np.asarray(
            handle["jets"].fields("eventNumber")[:], dtype=np.int64
        )
        official_eventwise_rows = int(handle["eventwise"].shape[0])
        eligible, protocol_audit = _protocol_eligible_mask(
            labels_all,
            pt_all,
            eta_all,
            events_all,
            protocol=protocol,
            max_source_events=max_source_events,
            random_state=random_state,
        )
        rows = _balanced_source_rows(
            labels_all, sample, random_state, eligible_mask=eligible
        )
        jets = _read_rows_chunked(handle["jets"], rows)

    raw_labels = np.asarray(jets[label_field], dtype=int)
    if set(np.unique(raw_labels).tolist()) != {LIGHT_LABEL, BOTTOM_LABEL}:
        raise RuntimeError("Balanced JetSet sample did not contain light and b labels.")
    y = (raw_labels == BOTTOM_LABEL).astype(int)
    events = np.asarray(jets["eventNumber"], dtype=np.int64)
    audit_scores = {
        "GN2v01_b_vs_light": _binary_audit_score(
            jets["GN2v01_pb"], jets["GN2v01_pu"]
        ),
        "DL1dv01_b_vs_light": _binary_audit_score(
            jets["DL1dv01_pb"], jets["DL1dv01_pu"]
        ),
        "GN2v01_paper_Db": gn2_paper_b_discriminant(
            jets["GN2v01_pb"],
            jets["GN2v01_pc"],
            jets["GN2v01_ptau"],
            jets["GN2v01_pu"],
        ),
        "DL1dv01_paper_Db": dl1d_paper_b_discriminant(
            jets["DL1dv01_pb"], jets["DL1dv01_pc"], jets["DL1dv01_pu"]
        ),
    }
    audit_variables = {
        "jet_pt_btagJes_MeV": np.asarray(jets["pt_btagJes"], dtype=float),
        "jet_eta_btagJes": np.asarray(jets["eta_btagJes"], dtype=float),
        "raw_truth_label_audit_only": raw_labels,
    }
    schema.update(
        {
            "official_eventwise_rows": official_eventwise_rows,
            "downloaded_official_file_contains_fewer_than_two_million_events": bool(
                official_eventwise_rows < 2000000
            ),
            "protocol_audit": protocol_audit,
            "selected_source_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "selected_jets": int(len(rows)),
            "selected_unique_events": int(len(np.unique(events))),
            "structured_tracks_materialized_at_index_stage": False,
            "GN2v01_vertexIndex_used_as_input": False,
        }
    )
    return AtlasJetSetIndex(
        y=y,
        row_index=rows,
        group_id=events,
        dataset_name=(
            "ATLAS JetSet ttbar 13.6 TeV %s: b=1 versus light=0" % protocol
        ),
        audit_scores=audit_scores,
        audit_variables=audit_variables,
        label_field=label_field,
        schema_audit=schema,
    )


class H5StructuredTrackSource:
    """Batch reader mapping logical sample positions to official HDF5 rows."""

    def __init__(self, path, selected_source_rows, max_tracks=20):
        self.path = str(Path(path))
        self.selected_source_rows = np.asarray(selected_source_rows, dtype=np.int64)
        self.max_tracks = int(max_tracks)
        if not 1 <= self.max_tracks <= 40:
            raise ValueError("max_tracks must lie in [1, 40].")
        self._handle = None

    def __enter__(self):
        self._handle = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("H5StructuredTrackSource must be used as a context manager.")

    def read(self, logical_indices):
        """Return transformed aligned tracks, mask, and reconstructed jet context."""
        self._require_open()
        logical_indices = np.asarray(logical_indices, dtype=int)
        if logical_indices.ndim != 1 or len(np.unique(logical_indices)) != len(
            logical_indices
        ):
            raise ValueError("Batch logical indices must be a unique one-dimensional array.")
        rows = self.selected_source_rows[logical_indices]
        order = np.argsort(rows)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        tracks = _read_rows_chunked(self._handle["tracks"], rows[order])[inverse]
        jets = _read_rows_chunked(self._handle["jets"], rows[order])[inverse]

        valid_all = np.asarray(tracks["valid"], dtype=bool)
        ranking = np.abs(
            np.nan_to_num(
                np.asarray(
                    tracks["lifetimeSignedD0Significance"], dtype=np.float32
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        ranking = np.where(valid_all, ranking, -np.inf)
        ranked = np.argsort(-ranking, axis=1, kind="stable")[:, : self.max_tracks]
        mask = np.take_along_axis(valid_all, ranked, axis=1)
        columns = []
        for name in RECONSTRUCTED_TRACK_INPUT_FIELDS:
            values = _track_transform(name, tracks[name]).astype(np.float32)
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

    def read_many(self, logical_indices, batch_size=2048):
        logical_indices = np.asarray(logical_indices, dtype=int)
        track_parts, mask_parts, jet_parts = [], [], []
        for start in range(0, len(logical_indices), int(batch_size)):
            values = self.read(logical_indices[start : start + int(batch_size)])
            track_parts.append(values[0])
            mask_parts.append(values[1])
            jet_parts.append(values[2])
        return (
            np.concatenate(track_parts, axis=0),
            np.concatenate(mask_parts, axis=0),
            np.concatenate(jet_parts, axis=0),
        )


class StructuredTrackScaler:
    """Base-training-only robust scaling for aligned nodes and jet context."""

    def __init__(self, clip=10.0):
        self.clip = float(clip)

    @staticmethod
    def _center_scale(values):
        center = np.median(values, axis=0)
        low = np.quantile(values, 0.10, axis=0)
        high = np.quantile(values, 0.90, axis=0)
        scale = np.maximum(high - low, 1e-6)
        return center.astype(np.float32), scale.astype(np.float32)

    def fit(self, tracks, mask, jets):
        tracks = np.asarray(tracks, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        jets = np.asarray(jets, dtype=np.float32)
        if tracks.ndim != 3 or tracks.shape[:2] != mask.shape:
            raise ValueError("Track tensor and mask shapes disagree.")
        if not np.any(mask):
            raise ValueError("Scaler received no valid tracks.")
        self.track_center_, self.track_scale_ = self._center_scale(tracks[mask])
        self.jet_center_, self.jet_scale_ = self._center_scale(jets)
        return self

    def transform(self, tracks, mask, jets):
        if not hasattr(self, "track_center_"):
            raise RuntimeError("StructuredTrackScaler must be fit first.")
        mask = np.asarray(mask, dtype=bool)
        track_scaled = (np.asarray(tracks, dtype=np.float32) - self.track_center_) / (
            self.track_scale_
        )
        jet_scaled = (np.asarray(jets, dtype=np.float32) - self.jet_center_) / (
            self.jet_scale_
        )
        track_scaled = np.clip(track_scaled, -self.clip, self.clip)
        jet_scaled = np.clip(jet_scaled, -self.clip, self.clip)
        track_scaled *= mask[:, :, None]
        if not np.isfinite(track_scaled).all() or not np.isfinite(jet_scaled).all():
            raise RuntimeError("Structured track scaling produced non-finite values.")
        return track_scaled.astype(np.float32), mask, jet_scaled.astype(np.float32)

    def summary(self):
        return {
            "fit_scope": "encoder-training events only",
            "track_fields": list(RECONSTRUCTED_TRACK_INPUT_FIELDS),
            "track_dimension": int(len(RECONSTRUCTED_TRACK_INPUT_FIELDS)),
            "jet_context_dimension": 4,
            "transform": "physics maps followed by median/(q90-q10) scaling",
            "clip": float(self.clip),
        }


class StructuredTrackGeometry:
    """Small shared-node attention encoder trained only as a geometry map.

    The implementation uses NumPy/BLAS so the workspace keeps the same light
    dependency stack as Unified V9.  Its auxiliary sigmoid exists solely to
    train the geometry; only the penultimate geometry enters POR.
    """

    def __init__(
        self,
        representation="set",
        hidden_dimension=24,
        geometry_dimension=48,
        max_tracks=20,
        graph_neighbors=6,
        sequential_steps=3,
        parallel_temperature=2.0,
        parallel_floor=0.02,
        learning_rate=2e-3,
        l2=1e-4,
        batch_size=512,
        max_epochs=12,
        patience=3,
        random_state=42,
    ):
        self.representation = str(representation)
        self.hidden_dimension = int(hidden_dimension)
        self.geometry_dimension = int(geometry_dimension)
        self.max_tracks = int(max_tracks)
        self.graph_neighbors = int(graph_neighbors)
        self.sequential_steps = int(sequential_steps)
        self.parallel_temperature = float(parallel_temperature)
        self.parallel_floor = float(parallel_floor)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        if self.representation not in STRUCTURED_REPRESENTATIONS:
            raise ValueError(
                "representation must be set, graph, sequential, parallel, or hybrid."
            )
        if self.hidden_dimension < 4 or self.geometry_dimension < 4:
            raise ValueError("Encoder dimensions must be at least four.")
        if self.sequential_steps < 1 or self.sequential_steps > 6:
            raise ValueError("sequential_steps must lie in [1, 6].")
        if self.parallel_temperature <= 0:
            raise ValueError("parallel_temperature must be positive.")
        if self.parallel_floor < 0:
            raise ValueError("parallel_floor cannot be negative.")

    @staticmethod
    def _sigmoid(values):
        values = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _relation_indices():
        return [
            RECONSTRUCTED_TRACK_INPUT_FIELDS.index(name)
            for name in (
                "deta",
                "dphi",
                "lifetimeSignedD0Significance",
                "lifetimeSignedZ0SinThetaSignificance",
            )
        ]

    def _relation_distance(self, tracks, mask):
        indices = [
            int(value) for value in self._relation_indices()
        ]
        relation = tracks[:, :, indices]
        difference = relation[:, :, None, :] - relation[:, None, :, :]
        distance = np.sum(difference * difference, axis=3)
        pair_valid = mask[:, :, None] & mask[:, None, :]
        diagonal = np.eye(mask.shape[1], dtype=bool)[None, :, :]
        distance = np.where(pair_valid & ~diagonal, distance, np.inf)
        return distance.astype(np.float32), pair_valid & ~diagonal

    def _local_adjacency(self, tracks, mask):
        distance, pair_valid = self._relation_distance(tracks, mask)
        k = min(max(1, self.graph_neighbors), max(1, mask.shape[1] - 1))
        nearest = np.argpartition(distance, kth=k - 1, axis=2)[:, :, :k]
        selected_distance = np.take_along_axis(distance, nearest, axis=2)
        weights = np.exp(-0.5 * np.minimum(selected_distance, 50.0))
        weights *= np.isfinite(selected_distance)
        adjacency = np.zeros_like(distance, dtype=np.float32)
        np.put_along_axis(adjacency, nearest, weights.astype(np.float32), axis=2)
        normalizer = np.sum(adjacency, axis=2, keepdims=True)
        adjacency = np.divide(
            adjacency,
            normalizer,
            out=np.zeros_like(adjacency),
            where=normalizer > 0,
        )
        adjacency *= pair_valid
        return adjacency.astype(np.float32)

    def _parallel_adjacency(self, tracks, mask):
        """Dense heat-kernel affinity for one-step region-to-all broadcast."""
        distance, pair_valid = self._relation_distance(tracks, mask)
        denominator = 2.0 * self.parallel_temperature * self.parallel_temperature
        finite_distance = np.where(pair_valid, distance, 0.0)
        affinity = np.exp(-np.minimum(finite_distance / denominator, 50.0))
        affinity = (affinity + self.parallel_floor) * pair_valid
        normalizer = np.sum(affinity, axis=2, keepdims=True)
        return np.divide(
            affinity,
            normalizer,
            out=np.zeros_like(affinity, dtype=np.float32),
            where=normalizer > 0,
        ).astype(np.float32)

    def _sequential_messages(self, adjacency, tracks):
        messages = []
        current = np.asarray(tracks, dtype=np.float32)
        for _ in range(self.sequential_steps):
            current = np.matmul(adjacency, current).astype(np.float32)
            messages.append(current)
        return messages

    def _node_input(self, tracks, mask):
        if self.representation == "set":
            return np.asarray(tracks, dtype=np.float32)
        if self.representation == "graph":
            local = np.matmul(self._local_adjacency(tracks, mask), tracks)
            return np.concatenate(
                [tracks, local, tracks - local, tracks * local], axis=2
            ).astype(np.float32)
        if self.representation == "sequential":
            sequential = self._sequential_messages(
                self._local_adjacency(tracks, mask), tracks
            )
            return np.concatenate([tracks] + sequential, axis=2).astype(np.float32)
        if self.representation == "parallel":
            broadcast = np.matmul(self._parallel_adjacency(tracks, mask), tracks)
            return np.concatenate(
                [tracks, broadcast, tracks - broadcast, tracks * broadcast],
                axis=2,
            ).astype(np.float32)
        if self.representation == "hybrid":
            sequential = self._sequential_messages(
                self._local_adjacency(tracks, mask), tracks
            )
            broadcast = np.matmul(self._parallel_adjacency(tracks, mask), tracks)
            return np.concatenate(
                [tracks] + sequential + [broadcast, tracks - broadcast], axis=2
            ).astype(np.float32)
        raise RuntimeError("Unknown structured representation.")

    def _initialize(self, node_dimension):
        rng = np.random.RandomState(self.random_state)

        def weight(rows, columns):
            limit = np.sqrt(6.0 / float(rows + columns))
            return rng.uniform(-limit, limit, size=(rows, columns)).astype(np.float32)

        pooled = 2 * self.hidden_dimension + 4
        self.params_ = {
            "Wn": weight(node_dimension, self.hidden_dimension),
            "bn": np.zeros(self.hidden_dimension, dtype=np.float32),
            "wa": weight(self.hidden_dimension, 1).ravel(),
            "Wg": weight(pooled, self.geometry_dimension),
            "bg": np.zeros(self.geometry_dimension, dtype=np.float32),
            "wo": weight(self.geometry_dimension, 1).ravel(),
            "bo": np.zeros(1, dtype=np.float32),
        }

    def _forward(self, tracks, mask, jets, cache=False):
        node_input = self._node_input(tracks, mask)
        pre_node = np.matmul(node_input, self.params_["Wn"]) + self.params_["bn"]
        node = np.tanh(pre_node).astype(np.float32)
        node *= mask[:, :, None]
        root_h = np.sqrt(float(self.hidden_dimension))
        attention_logits = np.matmul(node, self.params_["wa"]) / root_h
        attention_logits = np.where(mask, attention_logits, -1e9)
        shifted = attention_logits - np.max(attention_logits, axis=1, keepdims=True)
        exponent = np.exp(np.clip(shifted, -30.0, 0.0)) * mask
        attention = np.divide(
            exponent,
            np.sum(exponent, axis=1, keepdims=True),
            out=np.zeros_like(exponent, dtype=np.float32),
            where=np.sum(exponent, axis=1, keepdims=True) > 0,
        ).astype(np.float32)
        pooled_attention = np.sum(attention[:, :, None] * node, axis=1)
        counts = np.maximum(np.sum(mask, axis=1, keepdims=True), 1)
        pooled_mean = np.sum(node, axis=1) / counts
        pooled = np.concatenate([pooled_attention, pooled_mean, jets], axis=1)
        pre_geometry = np.matmul(pooled, self.params_["Wg"]) + self.params_["bg"]
        geometry = np.tanh(pre_geometry).astype(np.float32)
        logits = np.matmul(geometry, self.params_["wo"]) + self.params_["bo"][0]
        probability = self._sigmoid(logits)
        if not cache:
            return geometry, probability
        return geometry, probability, {
            "node_input": node_input,
            "node": node,
            "attention": attention,
            "counts": counts.astype(np.float32),
            "pooled": pooled,
        }

    def _loss_and_gradients(self, tracks, mask, jets, labels):
        geometry, probability, cache = self._forward(
            tracks, mask, jets, cache=True
        )
        labels = np.asarray(labels, dtype=np.float32)
        batch = float(len(labels))
        eps = 1e-7
        loss = -np.mean(
            labels * np.log(np.clip(probability, eps, 1.0))
            + (1.0 - labels) * np.log(np.clip(1.0 - probability, eps, 1.0))
        )
        for name in ("Wn", "wa", "Wg", "wo"):
            loss += 0.5 * self.l2 * float(np.sum(self.params_[name] ** 2))

        d_logit = (probability - labels) / batch
        gradients = {}
        gradients["wo"] = np.matmul(geometry.T, d_logit) + self.l2 * self.params_["wo"]
        gradients["bo"] = np.asarray([np.sum(d_logit)], dtype=np.float32)
        d_geometry = d_logit[:, None] * self.params_["wo"][None, :]
        d_pre_geometry = d_geometry * (1.0 - geometry * geometry)
        gradients["Wg"] = (
            np.matmul(cache["pooled"].T, d_pre_geometry)
            + self.l2 * self.params_["Wg"]
        )
        gradients["bg"] = np.sum(d_pre_geometry, axis=0)
        d_pooled = np.matmul(d_pre_geometry, self.params_["Wg"].T)

        hidden = self.hidden_dimension
        d_attention_pool = d_pooled[:, :hidden]
        d_mean_pool = d_pooled[:, hidden : 2 * hidden]
        node = cache["node"]
        attention = cache["attention"]
        d_node = attention[:, :, None] * d_attention_pool[:, None, :]
        d_node += (
            d_mean_pool[:, None, :] / cache["counts"][:, :, None]
        ) * mask[:, :, None]
        d_attention = np.sum(d_attention_pool[:, None, :] * node, axis=2)
        centered = d_attention - np.sum(d_attention * attention, axis=1, keepdims=True)
        d_attention_logits = attention * centered
        root_h = np.sqrt(float(hidden))
        gradients["wa"] = (
            np.sum(node * d_attention_logits[:, :, None], axis=(0, 1)) / root_h
            + self.l2 * self.params_["wa"]
        )
        d_node += (
            d_attention_logits[:, :, None]
            * self.params_["wa"][None, None, :]
            / root_h
        )
        d_pre_node = d_node * (1.0 - node * node) * mask[:, :, None]
        gradients["Wn"] = (
            np.einsum("bti,btj->ij", cache["node_input"], d_pre_node)
            + self.l2 * self.params_["Wn"]
        )
        gradients["bn"] = np.sum(d_pre_node, axis=(0, 1))
        return float(loss), gradients

    @staticmethod
    def _validation_record(labels, probability):
        return {
            "validation_log_loss": float(log_loss(labels, probability, labels=[0, 1])),
            "validation_auc": float(roc_auc_score(labels, probability)),
        }

    def fit(
        self,
        source,
        train_indices,
        train_labels,
        validation_indices,
        validation_labels,
        read_batch_size=4096,
    ):
        train_indices = np.asarray(train_indices, dtype=int)
        validation_indices = np.asarray(validation_indices, dtype=int)
        train_labels = np.asarray(train_labels, dtype=int)
        validation_labels = np.asarray(validation_labels, dtype=int)
        if set(np.unique(train_labels).tolist()) != {0, 1}:
            raise ValueError("Structured encoder training requires both classes.")
        if set(np.unique(validation_labels).tolist()) != {0, 1}:
            raise ValueError("Structured encoder validation requires both classes.")

        train_raw = source.read_many(train_indices, batch_size=read_batch_size)
        validation_raw = source.read_many(
            validation_indices, batch_size=read_batch_size
        )
        self.scaler_ = StructuredTrackScaler().fit(*train_raw)
        train = self.scaler_.transform(*train_raw)
        validation = self.scaler_.transform(*validation_raw)
        node_dimension = int(self._node_input(train[0][:1], train[1][:1]).shape[2])
        self._initialize(node_dimension)

        first = {name: np.zeros_like(value) for name, value in self.params_.items()}
        second = {name: np.zeros_like(value) for name, value in self.params_.items()}
        step = 0
        rng = np.random.RandomState(self.random_state + 19)
        best_params = None
        best_loss = np.inf
        best_epoch = 0
        stale = 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(train_indices))
            batch_losses = []
            for start in range(0, len(order), self.batch_size):
                chosen = order[start : start + self.batch_size]
                loss, gradients = self._loss_and_gradients(
                    train[0][chosen],
                    train[1][chosen],
                    train[2][chosen],
                    train_labels[chosen],
                )
                batch_losses.append(loss)
                step += 1
                for name in self.params_:
                    gradient = np.asarray(gradients[name], dtype=np.float32)
                    first[name] = 0.9 * first[name] + 0.1 * gradient
                    second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                    first_hat = first[name] / (1.0 - 0.9 ** step)
                    second_hat = second[name] / (1.0 - 0.999 ** step)
                    self.params_[name] -= self.learning_rate * first_hat / (
                        np.sqrt(second_hat) + 1e-8
                    )
            _, validation_probability = self._forward(*validation, cache=False)
            record = self._validation_record(
                validation_labels, validation_probability
            )
            record.update(
                {
                    "epoch": int(epoch),
                    "training_objective": float(np.mean(batch_losses)),
                }
            )
            self.history_.append(record)
            if record["validation_log_loss"] < best_loss - 1e-6:
                best_loss = record["validation_log_loss"]
                best_params = deepcopy(self.params_)
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_params is None:
            raise RuntimeError("Structured encoder did not produce a finite state.")
        self.params_ = best_params
        self.best_epoch_ = int(best_epoch)
        self.validation_log_loss_ = float(best_loss)
        best_record = self.history_[self.best_epoch_ - 1]
        self.validation_auc_ = float(best_record["validation_auc"])
        train_geometry, _ = self._forward(*train, cache=False)
        self.embedding_center_ = np.mean(train_geometry, axis=0).astype(np.float32)
        self.embedding_scale_ = np.maximum(
            np.std(train_geometry, axis=0), 1e-5
        ).astype(np.float32)
        self.node_input_dimension_ = node_dimension
        self.train_rows_ = int(len(train_indices))
        self.validation_rows_ = int(len(validation_indices))
        return self

    def transform_arrays(self, tracks, mask, jets):
        if not hasattr(self, "params_"):
            raise RuntimeError("StructuredTrackGeometry must be fit first.")
        scaled = self.scaler_.transform(tracks, mask, jets)
        geometry, auxiliary = self._forward(*scaled, cache=False)
        geometry = (geometry - self.embedding_center_) / self.embedding_scale_
        if not np.isfinite(geometry).all():
            raise RuntimeError("Structured track geometry produced non-finite values.")
        return geometry.astype(np.float32), auxiliary.astype(np.float32)

    def encode_source(self, source, logical_indices, batch_size=2048):
        logical_indices = np.asarray(logical_indices, dtype=int)
        geometry_parts = []
        auxiliary_parts = []
        for start in range(0, len(logical_indices), int(batch_size)):
            raw = source.read(logical_indices[start : start + int(batch_size)])
            geometry, auxiliary = self.transform_arrays(*raw)
            geometry_parts.append(geometry)
            auxiliary_parts.append(auxiliary)
        return {
            "geometry": np.concatenate(geometry_parts, axis=0),
            "auxiliary_probability": np.concatenate(auxiliary_parts, axis=0),
        }

    def summary(self):
        return {
            "role": "training_only_structured_track_geometry",
            "representation": self.representation,
            "max_tracks": int(self.max_tracks),
            "raw_track_fields": list(RECONSTRUCTED_TRACK_INPUT_FIELDS),
            "shared_node_encoder": True,
            "node_input_dimension": int(self.node_input_dimension_),
            "node_hidden_dimension": int(self.hidden_dimension),
            "pooling": "learned attention plus masked mean",
            "fixed_track_graph_message_passing": bool(
                self.representation != "set"
            ),
            "graph_relation_fields": (
                [
                    "deta",
                    "dphi",
                    "lifetimeSignedD0Significance",
                    "lifetimeSignedZ0SinThetaSignificance",
                ]
                if self.representation != "set"
                else []
            ),
            "graph_neighbors": (
                int(self.graph_neighbors)
                if self.representation in {"graph", "sequential", "hybrid"}
                else 0
            ),
            "sequential_local_steps": (
                int(self.sequential_steps)
                if self.representation in {"sequential", "hybrid"}
                else 0
            ),
            "sequential_channels": (
                ["H"]
                + ["A^%d H" % step for step in range(1, self.sequential_steps + 1)]
                if self.representation in {"sequential", "hybrid"}
                else []
            ),
            "dense_parallel_broadcast": bool(
                self.representation in {"parallel", "hybrid"}
            ),
            "parallel_kernel": (
                "dense reconstructed-relation heat-kernel affinity plus weak global floor"
                if self.representation in {"parallel", "hybrid"}
                else None
            ),
            "parallel_temperature": (
                float(self.parallel_temperature)
                if self.representation in {"parallel", "hybrid"}
                else None
            ),
            "parallel_connection_floor": (
                float(self.parallel_floor)
                if self.representation in {"parallel", "hybrid"}
                else None
            ),
            "original_track_residual_channel_preserved": True,
            "hybrid_channel_fusion": (
                "concatenate original, A^k local paths, dense global broadcast, and residual difference before shared learned node map"
                if self.representation == "hybrid"
                else None
            ),
            "geometry_dimension": int(self.geometry_dimension),
            "encoder_training_rows": int(self.train_rows_),
            "encoder_validation_rows": int(self.validation_rows_),
            "best_epoch": int(self.best_epoch_),
            "validation_log_loss": float(self.validation_log_loss_),
            "validation_auc_audit": float(self.validation_auc_),
            "auxiliary_probability_used_in_final_prediction": False,
            "auxiliary_probability_fusion": False,
            "truth_track_fields_used": False,
            "GN2_or_DL1_scores_used": False,
            "GN2v01_vertexIndex_used": False,
            "scaler": self.scaler_.summary(),
        }


class IdentityFeatureMap:
    """Adapter for already standardized structured-track geometries."""

    def __init__(self, dimension):
        self.dimension = int(dimension)

    def fit(self, X):
        self.transform(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError("Structured geometry dimension changed.")
        if not np.isfinite(X).all():
            raise ValueError("Structured geometry must be finite.")
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def summary(self):
        return {
            "name": "identity_for_standardized_structured_track_geometry",
            "output_dimension": int(self.dimension),
            "labels_used": False,
        }
