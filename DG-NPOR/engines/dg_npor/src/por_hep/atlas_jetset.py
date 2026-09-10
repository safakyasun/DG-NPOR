"""Leakage-safe ATLAS JetSet access and fixed-dimensional track summaries.

The official HDF5 file stores variable-length reconstructed track information
in a padded ``(n_jets, 40)`` structured array.  Unified V9 expects a finite
matrix, so this module maps the exact reconstructed GN2 input fields to a
permutation-invariant vector of within-jet distribution summaries.  Truth
labels and precomputed tagger scores are never included in ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import warnings
import zlib

import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler


JETSET_RECORD_URL = "https://opendata.cern.ch/record/93940"
JETSET_BUCKET_URL = (
    "https://opendata.cern.ch/api/files/"
    "9db0fcdc-5b9d-4abc-be74-54166856f6ff"
)
JETSET_FILES = {
    "small": {
        "name": "mc-flavtag-ttbar-small.h5",
        "size": 3055139850,
        "adler32": "8d7e9098",
    },
    "medium": {
        "name": "mc-flavtag-ttbar-medium.h5",
        "size": 13935881976,
        "adler32": "e29be59c",
    },
    "large": {
        "name": "mc-flavtag-ttbar-large.h5",
        "size": 90797951148,
        "adler32": "61adcbcb",
    },
}

LABEL_FIELDS = {
    "cone": "HadronConeExclTruthLabelID",
    "ghost": "HadronGhostTruthLabelID",
}
LIGHT_LABEL = 0
BOTTOM_LABEL = 5
JETSET_PROTOCOLS = ("pilot-b-light", "paper-ttbar")
PAPER_TTBAR_PT_MIN_MEV = 20000.0
PAPER_TTBAR_PT_MAX_MEV = 250000.0
PAPER_TTBAR_ABS_ETA_MAX = 2.5
PAPER_GN2_FC = 0.20
PAPER_GN2_FTAU = 0.05
PAPER_DL1D_FC = 0.018

# This is the reconstructed low-level input set documented in the public GN2
# open-data configuration.  The fields below are the only HDF5 fields allowed
# to contribute to the POR matrix.
RECONSTRUCTED_JET_INPUT_FIELDS = ("pt_btagJes", "eta_btagJes")
RECONSTRUCTED_TRACK_INPUT_FIELDS = (
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
TRACK_STATISTICS = ("mean", "std", "q10", "q25", "q50", "q75", "q90")

FORBIDDEN_JET_INPUT_FIELDS = {
    "eventNumber",
    "isJvtHS",
    "isJvtPU",
    "matchedToTruthJet",
    "HadronConeExclTruthLabelID",
    "HadronConeExclExtendedTruthLabelID",
    "HadronGhostTruthLabelID",
    "HadronGhostExtendedTruthLabelID",
    "PartonTruthLabelID",
    "PartonTruthLabelDR",
    "PartonTruthLabelPt",
    "ptFromTruthJet",
    "etaFromTruthJet",
    "phiFromTruthJet",
    "mFromTruthJet",
    "deltaEtaToTruthJet",
    "deltaPhiToTruthJet",
    "GN2v01_pb",
    "GN2v01_pc",
    "GN2v01_pu",
    "GN2v01_ptau",
    "DL1dv01_pb",
    "DL1dv01_pc",
    "DL1dv01_pu",
}
FORBIDDEN_TRACK_INPUT_FIELDS = {
    "ftagTruthParentBarcode",
    "ftagTruthOriginLabel",
    "ftagTruthVertexIndex",
    "GN2v01_trackOrigin",
    "GN2v01_vertexIndex",
}


@dataclass(frozen=True)
class AtlasJetSetDataset:
    X: np.ndarray
    y: np.ndarray
    row_index: np.ndarray
    feature_names: list
    dataset_name: str
    group_id: np.ndarray
    grouping_source: str
    audit_scores: dict
    audit_variables: dict
    label_field: str
    schema_audit: dict


def official_file_url(tier="small"):
    info = JETSET_FILES[str(tier)]
    return "%s/%s" % (JETSET_BUCKET_URL, info["name"])


def infer_official_tier(path):
    name = Path(path).name
    for tier, info in JETSET_FILES.items():
        if name == info["name"]:
            return tier
    return None


def file_adler32(path, chunk_size=16 * 1024 * 1024):
    value = 1
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            value = zlib.adler32(block, value)
    return "%08x" % (value & 0xFFFFFFFF)


def verify_official_file(path, tier=None):
    path = Path(path)
    tier = str(tier) if tier is not None else infer_official_tier(path)
    if tier not in JETSET_FILES:
        raise ValueError("Cannot infer the official JetSet tier from %s." % path.name)
    expected = JETSET_FILES[tier]
    if not path.exists():
        raise FileNotFoundError(path)
    size = int(path.stat().st_size)
    if size != int(expected["size"]):
        raise RuntimeError(
            "JetSet file size mismatch: expected %d, found %d." % (expected["size"], size)
        )
    checksum = file_adler32(path)
    if checksum != expected["adler32"]:
        raise RuntimeError(
            "JetSet Adler-32 mismatch: expected %s, found %s."
            % (expected["adler32"], checksum)
        )
    return {
        "tier": tier,
        "file_name": path.name,
        "size_bytes": size,
        "adler32": checksum,
        "verified_against_cern_record_93940": True,
    }


def download_official_file(destination, tier="small"):
    """Download one official JetSet tier, resume interruptions, and verify it.

    The in-progress file is deliberately retained as ``*.part``.  Re-running
    either entry point resumes that file instead of restarting a multi-GiB
    transfer from byte zero.
    """
    tier = str(tier)
    if tier not in JETSET_FILES:
        raise ValueError("tier must be small, medium, or large.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verify_official_file(destination, tier=tier)

    temporary = destination.with_name(destination.name + ".part")
    url = official_file_url(tier)
    expected_size = int(JETSET_FILES[tier]["size"])
    partial_size = int(temporary.stat().st_size) if temporary.exists() else 0
    if partial_size > expected_size:
        raise RuntimeError(
            "Partial JetSet download is larger than the official file: %s "
            "(%d > %d bytes). Move it aside and retry."
            % (temporary, partial_size, expected_size)
        )
    if partial_size == expected_size:
        result = verify_official_file(temporary, tier=tier)
        os.replace(str(temporary), str(destination))
        result["file_name"] = destination.name
        result["resumed_from_bytes"] = partial_size
        return result

    print("Downloading official ATLAS JetSet %s tier (%s)" % (tier, url))
    if partial_size:
        print(
            "Resuming preserved partial download at byte %d (%.1f%% complete)."
            % (partial_size, 100.0 * partial_size / expected_size)
        )
    curl = shutil.which("curl")
    if curl:
        command = [
            curl,
            "--location",
            "--fail",
            "--show-error",
            "--progress-bar",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--retry-connrefused",
            url,
            "--output",
            str(temporary),
        ]
        last_error = None
        for attempt in range(1, 13):
            try:
                subprocess.run(command, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                if attempt == 12:
                    break
                current_size = (
                    int(temporary.stat().st_size) if temporary.exists() else 0
                )
                delay = min(5 * attempt, 30)
                print(
                    "Transfer interrupted (curl exit %d); preserved %d bytes. "
                    "Retrying in %d seconds (%d/12)."
                    % (error.returncode, current_size, delay, attempt)
                )
                time.sleep(delay)
        if last_error is not None:
            current_size = int(temporary.stat().st_size) if temporary.exists() else 0
            raise RuntimeError(
                "JetSet download is still incomplete after automatic retries. "
                "The partial file was preserved at %s (%d bytes). Re-run the "
                "same command to resume it."
                % (temporary, current_size)
            ) from last_error
    else:
        last_error = None
        for attempt in range(1, 13):
            offset = int(temporary.stat().st_size) if temporary.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", "bytes=%d-" % offset)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = getattr(response, "status", response.getcode())
                    append = bool(offset and status == 206)
                    mode = "ab" if append else "wb"
                    copied = offset if append else 0
                    next_report = ((copied // (256 * 1024 * 1024)) + 1) * (
                        256 * 1024 * 1024
                    )
                    with temporary.open(mode) as handle:
                        while True:
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                            copied += len(block)
                            if copied >= next_report:
                                print("Downloaded %.2f GiB" % (copied / 1024.0**3))
                                next_report += 256 * 1024 * 1024
                last_error = None
                break
            except (OSError, urllib.error.URLError) as error:
                last_error = error
                if attempt == 12:
                    break
                current_size = (
                    int(temporary.stat().st_size) if temporary.exists() else 0
                )
                delay = min(5 * attempt, 30)
                print(
                    "Transfer interrupted; preserved %d bytes. Retrying in %d "
                    "seconds (%d/12)." % (current_size, delay, attempt)
                )
                time.sleep(delay)
        if last_error is not None:
            current_size = int(temporary.stat().st_size) if temporary.exists() else 0
            raise RuntimeError(
                "JetSet download is still incomplete after automatic retries. "
                "The partial file was preserved at %s (%d bytes). Re-run the "
                "same command to resume it."
                % (temporary, current_size)
            ) from last_error
    result = verify_official_file(temporary, tier=tier)
    os.replace(str(temporary), str(destination))
    result["file_name"] = destination.name
    result["resumed_from_bytes"] = partial_size
    return result


def _required_schema(label_field):
    return {
        "jets": set(RECONSTRUCTED_JET_INPUT_FIELDS)
        | {
            label_field,
            "eventNumber",
            "GN2v01_pb",
            "GN2v01_pc",
            "GN2v01_pu",
            "GN2v01_ptau",
            "DL1dv01_pb",
            "DL1dv01_pc",
            "DL1dv01_pu",
        },
        "tracks": set(RECONSTRUCTED_TRACK_INPUT_FIELDS) | {"valid"},
        "eventwise": set(),
        "truth_hadrons": set(),
    }


def inspect_schema(path, label_definition="cone"):
    label_field = LABEL_FIELDS[str(label_definition)]
    required = _required_schema(label_field)
    with h5py.File(path, "r") as handle:
        missing_datasets = sorted(set(required) - set(handle.keys()))
        if missing_datasets:
            raise ValueError("JetSet datasets are missing: %s" % missing_datasets)
        shapes = {name: list(handle[name].shape) for name in required}
        missing_fields = {}
        for name, fields in required.items():
            available = set(handle[name].dtype.names or ())
            absent = sorted(fields - available)
            if absent:
                missing_fields[name] = absent
        if missing_fields:
            raise ValueError("JetSet fields are missing: %s" % missing_fields)
        if handle["tracks"].shape[0] != handle["jets"].shape[0]:
            raise ValueError("Jet and track row counts differ.")
        if handle["tracks"].shape[1] != 40:
            raise ValueError("Expected up to 40 tracks per JetSet jet.")

    input_jet = set(RECONSTRUCTED_JET_INPUT_FIELDS)
    input_track = set(RECONSTRUCTED_TRACK_INPUT_FIELDS)
    forbidden_overlap = sorted(
        (input_jet & FORBIDDEN_JET_INPUT_FIELDS)
        | (input_track & FORBIDDEN_TRACK_INPUT_FIELDS)
    )
    if forbidden_overlap:
        raise RuntimeError("Forbidden truth/tagger fields entered the adapter.")
    return {
        "dataset_shapes": shapes,
        "label_definition": str(label_definition),
        "truth_label_field_audit_only": label_field,
        "reconstructed_jet_input_fields": list(RECONSTRUCTED_JET_INPUT_FIELDS),
        "reconstructed_track_input_fields": list(RECONSTRUCTED_TRACK_INPUT_FIELDS),
        "truth_hadrons_dataset_used_as_input": False,
        "eventwise_truth_displacements_used_as_input": False,
        "GN2_or_DL1_scores_used_as_input": False,
        "forbidden_input_overlap": forbidden_overlap,
        "track_order_dependence": False,
        "track_summary_statistics": list(TRACK_STATISTICS),
    }


def _read_rows_chunked(dataset, rows):
    """Read sorted HDF5 rows once per compressed chunk."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("HDF5 source rows must be strictly increasing.")
    output = np.empty((len(rows),) + dataset.shape[1:], dtype=dataset.dtype)
    chunk = int(dataset.chunks[0] if dataset.chunks else 4096)
    chunk_ids = rows // chunk
    for chunk_id in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids == chunk_id)
        start = int(chunk_id * chunk)
        stop = min(start + chunk, int(dataset.shape[0]))
        block = dataset[start:stop]
        output[positions] = block[rows[positions] - start]
    return output


def _balanced_source_rows(labels, sample, random_state, eligible_mask=None):
    labels = np.asarray(labels, dtype=int)
    eligible = (
        np.ones(len(labels), dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    if len(eligible) != len(labels):
        raise ValueError("Eligible mask and label array lengths differ.")
    light = np.flatnonzero(eligible & (labels == LIGHT_LABEL))
    bottom = np.flatnonzero(eligible & (labels == BOTTOM_LABEL))
    if int(sample) <= 0:
        per_class = min(len(light), len(bottom))
    else:
        if int(sample) % 2:
            raise ValueError("--sample must be even for exact b/light balance.")
        per_class = int(sample) // 2
    if per_class > min(len(light), len(bottom)):
        raise ValueError("Requested balanced sample exceeds the available minority class.")
    rng = np.random.RandomState(int(random_state))
    chosen_light = rng.choice(light, size=per_class, replace=False)
    chosen_bottom = rng.choice(bottom, size=per_class, replace=False)
    return np.sort(np.concatenate([chosen_light, chosen_bottom])).astype(np.int64)


def _protocol_eligible_mask(
    labels,
    pt_mev,
    eta,
    event_numbers,
    protocol,
    max_source_events,
    random_state,
):
    """Build a label-blind event cap followed by protocol physics cuts."""
    protocol = str(protocol)
    if protocol not in JETSET_PROTOCOLS:
        raise ValueError("protocol must be one of %s." % (JETSET_PROTOCOLS,))
    labels = np.asarray(labels, dtype=int)
    pt_mev = np.asarray(pt_mev, dtype=float)
    eta = np.asarray(eta, dtype=float)
    event_numbers = np.asarray(event_numbers, dtype=np.int64)
    unique_events = np.unique(event_numbers)
    requested_limit = int(max_source_events)
    if requested_limit < 0:
        raise ValueError("max_source_events must be zero or positive.")
    event_cap_binding = bool(requested_limit and len(unique_events) > requested_limit)
    if event_cap_binding:
        rng = np.random.RandomState(int(random_state) + 1701)
        retained_events = rng.choice(
            unique_events, size=requested_limit, replace=False
        )
        event_mask = np.isin(event_numbers, retained_events)
        retained_event_count = requested_limit
    else:
        event_mask = np.ones(len(labels), dtype=bool)
        retained_event_count = int(len(unique_events))

    finite_kinematics = np.isfinite(pt_mev) & np.isfinite(eta)
    if protocol == "paper-ttbar":
        physics_mask = (
            finite_kinematics
            & (pt_mev > PAPER_TTBAR_PT_MIN_MEV)
            & (pt_mev < PAPER_TTBAR_PT_MAX_MEV)
            & (np.abs(eta) < PAPER_TTBAR_ABS_ETA_MAX)
        )
        cuts = {
            "jet_pt_btagJes_MeV": {
                "lower_exclusive": PAPER_TTBAR_PT_MIN_MEV,
                "upper_exclusive": PAPER_TTBAR_PT_MAX_MEV,
            },
            "abs_jet_eta_btagJes": {
                "upper_exclusive": PAPER_TTBAR_ABS_ETA_MAX,
            },
        }
    else:
        physics_mask = finite_kinematics
        cuts = {"finite_reconstructed_kinematics_only": True}

    binary_label_mask = (labels == LIGHT_LABEL) | (labels == BOTTOM_LABEL)
    eligible = event_mask & physics_mask & binary_label_mask
    return eligible, {
        "analysis_protocol": protocol,
        "requested_max_source_events": requested_limit,
        "available_unique_source_events": int(len(unique_events)),
        "source_event_cap_binding": event_cap_binding,
        "retained_source_events_before_physics_cuts": retained_event_count,
        "official_hdf5_is_monolithic_partial_event_download": False,
        "physics_cuts": cuts,
        "eligible_binary_jets_after_event_cap_and_cuts": int(np.sum(eligible)),
        "eligible_light_jets": int(np.sum(eligible & (labels == LIGHT_LABEL))),
        "eligible_bottom_jets": int(np.sum(eligible & (labels == BOTTOM_LABEL))),
    }


def _signed_log1p(values, scale=1.0):
    values = np.asarray(values, dtype=float) * float(scale)
    return np.sign(values) * np.log1p(np.abs(values))


def _track_transform(name, values):
    values = np.asarray(values, dtype=float)
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


def jetset_feature_names():
    names = [
        "jet_log1p_pt_btagJes_GeV",
        "jet_eta_btagJes",
        "jet_abs_eta_btagJes",
        "n_valid_tracks",
    ]
    for field in RECONSTRUCTED_TRACK_INPUT_FIELDS:
        for statistic in TRACK_STATISTICS:
            names.append("track_%s_%s" % (field, statistic))
    return names


def _summarize_reconstructed_inputs(jets, tracks):
    valid = np.asarray(tracks["valid"], dtype=bool)
    columns = [
        np.log1p(np.maximum(np.asarray(jets["pt_btagJes"], dtype=float) / 1000.0, 0.0)),
        np.asarray(jets["eta_btagJes"], dtype=float),
        np.abs(np.asarray(jets["eta_btagJes"], dtype=float)),
        valid.sum(axis=1).astype(float),
    ]
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    for name in RECONSTRUCTED_TRACK_INPUT_FIELDS:
        values = _track_transform(name, tracks[name])
        values = np.where(valid & np.isfinite(values), values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            statistics = [
                np.nanmean(values, axis=1),
                np.nanstd(values, axis=1),
            ]
            q_values = np.nanquantile(values, quantiles, axis=1)
        statistics.extend(q_values[index] for index in range(len(quantiles)))
        columns.extend(np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0) for column in statistics)
    X = np.column_stack(columns).astype(np.float32)
    if X.shape[1] != len(jetset_feature_names()):
        raise RuntimeError("JetSet adapter feature-width mismatch.")
    if not np.isfinite(X).all():
        raise RuntimeError("JetSet reconstructed adapter produced non-finite values.")
    return X


def _binary_audit_score(pb, pu):
    pb = np.asarray(pb, dtype=float)
    pu = np.asarray(pu, dtype=float)
    denominator = pb + pu
    score = np.divide(pb, denominator, out=np.full_like(pb, 0.5), where=denominator > 0)
    return np.clip(np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def gn2_paper_b_discriminant(pb, pc, ptau, pu):
    """GN2 b-tagging discriminant from Eq. (1) of the GN2 paper."""
    pb = np.asarray(pb, dtype=float)
    background = (
        PAPER_GN2_FC * np.asarray(pc, dtype=float)
        + PAPER_GN2_FTAU * np.asarray(ptau, dtype=float)
        + (1.0 - PAPER_GN2_FC - PAPER_GN2_FTAU) * np.asarray(pu, dtype=float)
    )
    return np.log(np.clip(pb, 1e-12, None) / np.clip(background, 1e-12, None))


def dl1d_paper_b_discriminant(pb, pc, pu):
    """DL1d b-tagging discriminant used as the GN2 paper baseline."""
    pb = np.asarray(pb, dtype=float)
    background = (
        PAPER_DL1D_FC * np.asarray(pc, dtype=float)
        + (1.0 - PAPER_DL1D_FC) * np.asarray(pu, dtype=float)
    )
    return np.log(np.clip(pb, 1e-12, None) / np.clip(background, 1e-12, None))


def load_atlas_jetset_b_vs_light(
    path,
    sample=20000,
    random_state=42,
    label_definition="cone",
    protocol="pilot-b-light",
    max_source_events=2000000,
):
    """Load an exactly balanced b/light sample with event grouping."""
    path = Path(path)
    label_field = LABEL_FIELDS[str(label_definition)]
    schema = inspect_schema(path, label_definition=label_definition)
    with h5py.File(path, "r") as handle:
        labels_all = np.asarray(handle["jets"].fields(label_field)[:], dtype=int)
        pt_all = np.asarray(handle["jets"].fields("pt_btagJes")[:], dtype=float)
        eta_all = np.asarray(handle["jets"].fields("eta_btagJes")[:], dtype=float)
        events_all = np.asarray(handle["jets"].fields("eventNumber")[:], dtype=np.int64)
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
        tracks = _read_rows_chunked(handle["tracks"], rows)

    raw_labels = np.asarray(jets[label_field], dtype=int)
    if set(np.unique(raw_labels).tolist()) != {LIGHT_LABEL, BOTTOM_LABEL}:
        raise RuntimeError("Balanced JetSet sample did not contain light and b labels.")
    y = (raw_labels == BOTTOM_LABEL).astype(int)
    X = _summarize_reconstructed_inputs(jets, tracks)
    events = np.asarray(jets["eventNumber"], dtype=np.int64)
    audit_scores = {
        "GN2v01_b_vs_light": _binary_audit_score(jets["GN2v01_pb"], jets["GN2v01_pu"]),
        "DL1dv01_b_vs_light": _binary_audit_score(jets["DL1dv01_pb"], jets["DL1dv01_pu"]),
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
            "engineered_feature_dimension": int(X.shape[1]),
        }
    )
    return AtlasJetSetDataset(
        X=X,
        y=y,
        row_index=rows,
        feature_names=jetset_feature_names(),
        dataset_name=(
            "ATLAS JetSet ttbar 13.6 TeV %s: b=1 versus light=0" % protocol
        ),
        group_id=events,
        grouping_source="jets.eventNumber",
        audit_scores=audit_scores,
        audit_variables=audit_variables,
        label_field=label_field,
        schema_audit=schema,
    )


class AtlasJetSetFeatureMap:
    """Base-role-only robust scaling of the leakage-safe 137-vector."""

    def __init__(self, feature_names=None, clip=12.0):
        self.feature_names = list(feature_names or jetset_feature_names())
        self.clip = float(clip)
        if self.feature_names != jetset_feature_names():
            raise ValueError("JetSet feature order differs from the locked protocol.")

    def fit(self, X):
        X = self._validate(X)
        self.scaler_ = RobustScaler(quantile_range=(10.0, 90.0)).fit(X)
        self.output_feature_names_ = list(self.feature_names)
        return self

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("JetSet matrix has the wrong feature width.")
        if not np.isfinite(X).all():
            raise ValueError("JetSet matrix must be finite.")
        return X

    def transform(self, X):
        if not hasattr(self, "scaler_"):
            raise RuntimeError("AtlasJetSetFeatureMap must be fit first.")
        transformed = self.scaler_.transform(self._validate(X))
        transformed = np.clip(transformed, -self.clip, self.clip)
        if not np.isfinite(transformed).all():
            raise RuntimeError("JetSet scaler produced non-finite values.")
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def summary(self):
        if not hasattr(self, "output_feature_names_"):
            raise RuntimeError("AtlasJetSetFeatureMap must be fit first.")
        return {
            "name": "atlas_jetset_gn2_reco_track_summaries_v1",
            "labels_used_by_adapter": False,
            "fit_scope": "base_training_role_only",
            "raw_reconstructed_jet_fields": list(RECONSTRUCTED_JET_INPUT_FIELDS),
            "raw_reconstructed_track_fields": list(RECONSTRUCTED_TRACK_INPUT_FIELDS),
            "track_summary_statistics": list(TRACK_STATISTICS),
            "raw_feature_dimension": len(self.feature_names),
            "output_dimension": len(self.feature_names),
            "transform": "physics unit/log maps, permutation-invariant track summaries, robust scaling",
            "truth_hadron_fields_in_X": False,
            "truth_track_fields_in_X": False,
            "GN2_or_DL1_scores_in_X": False,
            "PCA": False,
            "user_selected_bottleneck": False,
            "compact_dimension_is": "K_DG selected later by dJ/dgate",
        }
