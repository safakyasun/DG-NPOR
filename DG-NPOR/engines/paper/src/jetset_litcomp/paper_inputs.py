"""Strict loading and alignment for the paper result workspace."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import load_existing_locked_predictions, load_frozen_split


POR_METHOD = "DG NPOR"
PARTICLENET_METHOD = "ParticleNet matched input"
GN2_METHOD = "ATLAS GN2v01 audit"
DL1_METHOD = "ATLAS DL1dv01 audit"
KEY_COLUMNS = ("source_row", "event_number", "y_true_light0_b1")


@dataclass(frozen=True)
class ComparisonInputs:
    reference: pd.DataFrame
    main_scores: dict[str, np.ndarray]
    audit_scores: dict[str, np.ndarray]
    dimensions: dict[str, int]
    probability_methods: set[str]
    particle_net_manifest: dict


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_reference(frame: pd.DataFrame):
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Locked test contains duplicated alignment keys.")
    if not np.isfinite(frame["self_configuring_por_probability_b"].to_numpy(float)).all():
        raise ValueError("DG NPOR score contains non-finite values.")
    probability = frame["self_configuring_por_probability_b"].to_numpy(float)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("DG NPOR probability lies outside [0, 1].")


def load_comparison_inputs(selfconfig_output_dir, particle_net_output):
    """Load DG NPOR, matched ParticleNet, and same-jet audit scores."""
    reference = load_existing_locked_predictions(selfconfig_output_dir).copy()
    _validate_reference(reference)
    split = load_frozen_split(selfconfig_output_dir)
    expected = pd.DataFrame({
        "source_row": split.source_rows[split.independent_test],
        "event_number": split.event_numbers[split.independent_test],
    })
    if len(reference) != len(expected):
        raise ValueError("DG NPOR predictions do not contain exactly the frozen test rows.")
    reference = expected.merge(reference, on=["source_row", "event_number"], how="left", validate="one_to_one", sort=False)
    if reference.isna().any().any():
        raise ValueError("DG NPOR predictions differ from the frozen test row/event keys.")
    selection_path = Path(selfconfig_output_dir) / "selected_derivative_gated_dimension.json"
    if not selection_path.exists():
        raise FileNotFoundError("Missing frozen dimension metadata: " + str(selection_path))
    selection = _read_json(selection_path)
    dimension = int(selection.get("K_DG", selection.get("selected_K", 0)))
    if dimension < 1:
        raise ValueError("Invalid frozen DG NPOR dimension.")

    particle_dir = Path(particle_net_output)
    prediction_path = particle_dir / "particle_net_locked_test_predictions.csv.gz"
    manifest_path = particle_dir / "particle_net_manifest.json"
    if not prediction_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "ParticleNet output must contain particle_net_locked_test_predictions.csv.gz "
            "and particle_net_manifest.json."
        )
    particle = pd.read_csv(prediction_path)
    if len(particle) != len(reference) or particle["source_row"].duplicated().any():
        raise ValueError("ParticleNet must contain exactly the frozen test rows, without extras.")
    required = set(KEY_COLUMNS) | {"particle_net_probability_b"}
    missing = sorted(required - set(particle.columns))
    if missing:
        raise ValueError(f"ParticleNet predictions are missing columns: {missing}")
    if particle.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("ParticleNet predictions contain duplicated alignment keys.")
    aligned = reference[list(KEY_COLUMNS)].merge(
        particle[list(required)],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(reference) or aligned["particle_net_probability_b"].isna().any():
        raise RuntimeError("ParticleNet and DG NPOR locked test rows do not match exactly.")
    probability = aligned["particle_net_probability_b"].to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("ParticleNet probability is non-finite or outside [0, 1].")

    manifest = _read_json(manifest_path)
    architecture = str(manifest.get("architecture", ""))
    if architecture != "full":
        raise ValueError(
            "The paper comparison requires the full ParticleNet architecture. "
            f"Found architecture={architecture!r}."
        )
    model = manifest.get("model", {})
    particle_dimension = int(model.get("paper_dense_units", 0))
    if particle_dimension != 256:
        raise ValueError(
            "The full ParticleNet manifest must report a 256 dimensional final hidden layer."
        )

    main_scores = {
        POR_METHOD: reference["self_configuring_por_probability_b"].to_numpy(float),
        PARTICLENET_METHOD: probability,
    }
    audit_scores = {
        GN2_METHOD: reference["GN2v01_paper_Db_audit_only"].to_numpy(float),
        DL1_METHOD: reference["DL1dv01_paper_Db_audit_only"].to_numpy(float),
    }
    for method, values in audit_scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{method} contains non-finite values.")
    return ComparisonInputs(
        reference=reference,
        main_scores=main_scores,
        audit_scores=audit_scores,
        dimensions={POR_METHOD: dimension, PARTICLENET_METHOD: particle_dimension},
        probability_methods=set(main_scores),
        particle_net_manifest=manifest,
    )


ORBITAL_CANDIDATES = (
    "self_configuring_locked_test_orbitals.csv.gz",
    "self_configuring_locked_test_orbitals.csv",
    "self_configuring_locked_test_predictions_with_orbitals.csv.gz",
    "self_configuring_locked_test_predictions.csv.gz",
    "self_configuring_locked_test_orbitals.npz",
)


def discover_orbital_file(selfconfig_output_dir, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Orbital file not found: {path}")
        return path
    directory = Path(selfconfig_output_dir)
    paths = [directory / name for name in ORBITAL_CANDIDATES]
    for pattern in ("*locked*test*orbital*.csv*", "*locked*test*state*.csv*"):
        paths.extend(sorted(directory.glob(pattern)))
    paths = list(dict.fromkeys(paths))
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            return path
        header = pd.read_csv(path, nrows=2)
        try:
            identify_orbital_columns(header.columns)
            return path
        except ValueError:
            continue
    return None


def identify_orbital_columns(columns):
    """Read the complete selected state, without silently truncating it to K=3."""
    for prefix in ("normalized_orbital", "selected_orbital", "orbital", "phi", "psi"):
        pattern = re.compile(r"^" + prefix + r"_?([0-9]+)$", re.I)
        matches = []
        for column in columns:
            found = pattern.match(str(column))
            if found:
                matches.append((int(found.group(1)), column))
        if matches:
            return [column for _, column in sorted(matches)]
    raise ValueError("No selected orbital columns found.")


def _load_npz_orbitals(path: Path):
    with np.load(path) as values:
        required = set(KEY_COLUMNS)
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"Orbital NPZ is missing arrays: {missing}")
        if "orbitals" in values.files:
            orbitals = np.asarray(values["orbitals"], dtype=float)
        elif "normalized_state" in values.files:
            orbitals = np.asarray(values["normalized_state"], dtype=float)
        else:
            raise ValueError("Orbital NPZ needs an orbitals or normalized_state array.")
        if orbitals.ndim != 2 or orbitals.shape[1] < 1:
            raise ValueError("Orbital array must have shape (number of jets, K), K >= 1.")
        return pd.DataFrame(
            {
                "source_row": values["source_row"],
                "event_number": values["event_number"],
                "y_true_light0_b1": values["y_true_light0_b1"],
                **{f"orbital_{i + 1}": orbitals[:, i] for i in range(orbitals.shape[1])},
            }
        )


def load_aligned_orbitals(path, reference):
    """Load and strictly align the full selected locked test state."""
    path = Path(path)
    frame = _load_npz_orbitals(path) if path.suffix == ".npz" else pd.read_csv(path)
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Orbital file is missing alignment columns: {missing}")
    columns = identify_orbital_columns(frame.columns)
    selected = frame[list(KEY_COLUMNS) + columns].copy()
    canonical = [f"orbital_{i + 1}" for i in range(len(columns))]
    selected = selected.rename(columns=dict(zip(columns, canonical)))
    if len(selected) != len(reference):
        raise ValueError("Orbital export must contain exactly the frozen test rows.")
    if selected.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Orbital file contains duplicated alignment keys.")
    aligned = reference[list(KEY_COLUMNS)].merge(
        selected,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(reference) or aligned[canonical].isna().any().any():
        raise RuntimeError("Orbital values and locked test rows do not match exactly.")
    values = aligned[canonical].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Orbital values contain non-finite values.")
    return aligned
