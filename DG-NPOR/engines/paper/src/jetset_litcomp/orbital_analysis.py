"""Descriptive analysis of the three selected predictive orbitals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import JetSetBatchReader, TRACK_FIELDS


ORBITAL_COLUMNS = ("orbital_1", "orbital_2", "orbital_3")


FEATURE_GROUPS = {
    "Impact parameter": (
        "max_abs_d0_significance",
        "mean_top3_abs_d0_significance",
        "max_abs_z0_significance",
    ),
    "Track kinematics": (
        "track_angular_rms",
        "mean_abs_deta",
        "mean_abs_dphi",
        "mean_abs_q_over_p",
    ),
    "Hit structure": (
        "mean_pixel_hits",
        "mean_sct_hits",
        "mean_shared_hits",
        "mean_split_hits",
    ),
    "Jet context": (
        "jet_log_pt",
        "jet_abs_eta",
        "tracks_used",
        "tracks_total",
    ),
}


def _masked_mean(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    total = np.sum(np.where(mask, values, 0.0), axis=1)
    count = np.maximum(mask.sum(axis=1), 1)
    return total / count


def _masked_max(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    safe = np.where(mask, values, -np.inf)
    result = np.max(safe, axis=1)
    result[~np.isfinite(result)] = 0.0
    return result


def _batch_summaries(tracks, mask, context):
    index = {name: TRACK_FIELDS.index(name) for name in TRACK_FIELDS}
    d0 = np.abs(tracks[:, :, index["lifetimeSignedD0Significance"]])
    z0 = np.abs(tracks[:, :, index["lifetimeSignedZ0SinThetaSignificance"]])
    deta = tracks[:, :, index["deta"]]
    dphi = tracks[:, :, index["dphi"]]
    q_over_p = tracks[:, :, index["qOverP"]]
    pixel = tracks[:, :, index["numberOfPixelHits"]]
    sct = tracks[:, :, index["numberOfSCTHits"]]
    shared = (
        tracks[:, :, index["numberOfInnermostPixelLayerSharedHits"]]
        + tracks[:, :, index["numberOfPixelSharedHits"]]
        + tracks[:, :, index["numberOfSCTSharedHits"]]
    )
    split = (
        tracks[:, :, index["numberOfInnermostPixelLayerSplitHits"]]
        + tracks[:, :, index["numberOfPixelSplitHits"]]
    )
    top3_mask = mask.copy()
    if top3_mask.shape[1] > 3:
        top3_mask[:, 3:] = False
    angular = np.sqrt(np.square(deta) + np.square(dphi))
    return pd.DataFrame(
        {
            "max_abs_d0_significance": _masked_max(d0, mask),
            "mean_top3_abs_d0_significance": _masked_mean(d0, top3_mask),
            "max_abs_z0_significance": _masked_max(z0, mask),
            "track_angular_rms": np.sqrt(_masked_mean(np.square(angular), mask)),
            "mean_abs_deta": _masked_mean(np.abs(deta), mask),
            "mean_abs_dphi": _masked_mean(np.abs(dphi), mask),
            "mean_abs_q_over_p": _masked_mean(np.abs(q_over_p), mask),
            "mean_pixel_hits": _masked_mean(pixel, mask),
            "mean_sct_hits": _masked_mean(sct, mask),
            "mean_shared_hits": _masked_mean(shared, mask),
            "mean_split_hits": _masked_mean(split, mask),
            "jet_log_pt": context[:, 0],
            "jet_abs_eta": context[:, 2],
            "tracks_used": mask.sum(axis=1).astype(float),
            "tracks_total": context[:, 3],
        }
    )


def reconstructed_feature_summaries(
    h5_file,
    reference,
    batch_size=2048,
    max_tracks=20,
):
    """Read the matched reconstructed inputs and form auditable jet summaries."""
    path = Path(h5_file)
    if not path.exists():
        raise FileNotFoundError(f"ATLAS JetSet HDF5 file not found: {path}")
    source_rows = reference["source_row"].to_numpy(np.int64)
    expected_labels = reference["y_true_light0_b1"].to_numpy(int)
    expected_events = reference["event_number"].to_numpy(np.int64)
    order = np.argsort(source_rows, kind="stable")
    sorted_source = source_rows[order]
    sorted_labels = expected_labels[order]
    sorted_events = expected_events[order]
    if len(sorted_source) and np.any(np.diff(sorted_source) <= 0):
        raise ValueError("Locked test source rows must be unique.")
    frames = []
    with JetSetBatchReader(path, sorted_source, max_tracks=max_tracks) as reader:
        labels, events = reader.read_labels_and_events()
        if not np.array_equal(labels, sorted_labels):
            raise RuntimeError("HDF5 labels do not match the locked test export.")
        if not np.array_equal(events, sorted_events):
            raise RuntimeError("HDF5 event numbers do not match the locked test export.")
        logical = np.arange(len(reference), dtype=int)
        for start in range(0, len(logical), int(batch_size)):
            batch = logical[start : start + int(batch_size)]
            tracks, mask, context = reader.read(batch)
            frames.append(_batch_summaries(tracks, mask, context))
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "source_row", sorted_source)
    result.insert(1, "event_number", sorted_events)
    result.insert(2, "y_true_light0_b1", sorted_labels)
    if not np.isfinite(result.iloc[:, 3:].to_numpy(float)).all():
        raise RuntimeError("Reconstructed feature summaries contain non-finite values.")
    keys = ["source_row", "event_number", "y_true_light0_b1"]
    return reference[keys].merge(
        result,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )


def analyze_orbitals(orbital_frame, feature_frame):
    """Return class summaries and feature associations without causal claims."""
    from .paper_inputs import identify_orbital_columns
    orbital_columns = identify_orbital_columns(orbital_frame.columns)
    keys = ["source_row", "event_number", "y_true_light0_b1"]
    merged = orbital_frame[keys + orbital_columns].merge(
        feature_frame,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(orbital_frame):
        raise RuntimeError("Orbital and reconstructed feature rows are not identical.")

    class_rows = []
    for orbital in orbital_columns:
        for label, class_name in ((0, "light"), (1, "bottom")):
            values = merged.loc[merged["y_true_light0_b1"] == label, orbital].to_numpy(float)
            class_rows.append(
                {
                    "orbital": orbital,
                    "class": class_name,
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "quantile_16": float(np.quantile(values, 0.16)),
                    "quantile_84": float(np.quantile(values, 0.84)),
                }
            )

    feature_columns = [feature for values in FEATURE_GROUPS.values() for feature in values]
    correlation = merged[orbital_columns + feature_columns].corr(method="spearman")
    long_rows = []
    group_lookup = {
        feature: group for group, features in FEATURE_GROUPS.items() for feature in features
    }
    for orbital in orbital_columns:
        for feature in feature_columns:
            long_rows.append(
                {
                    "orbital": orbital,
                    "feature_group": group_lookup[feature],
                    "feature": feature,
                    "spearman_rho": float(correlation.loc[orbital, feature]),
                    "absolute_spearman_rho": float(abs(correlation.loc[orbital, feature])),
                }
            )
    long = pd.DataFrame(long_rows)
    representatives = (
        long.sort_values("absolute_spearman_rho", ascending=False)
        .groupby(["orbital", "feature_group"], as_index=False, sort=False)
        .first()
    )
    return {
        "orbital_columns": orbital_columns,
        "merged": merged,
        "class_summary": pd.DataFrame(class_rows),
        "feature_correlations": long,
        "group_representatives": representatives,
    }
