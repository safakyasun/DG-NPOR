import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from jetset_litcomp.data import LABEL_FIELD, TRACK_FIELDS
from jetset_litcomp.paper_inputs import (
    PARTICLENET_METHOD,
    POR_METHOD,
    load_aligned_orbitals,
    load_comparison_inputs,
)


def _make_fixture(root: Path):
    selfconfig = root / "selfconfig"
    particle = root / "particle"
    selfconfig.mkdir()
    particle.mkdir()
    n_rows = 60
    source = np.arange(n_rows, 2 * n_rows, dtype=np.int64)
    events = np.repeat(np.arange(n_rows // 2, n_rows, dtype=np.int64), 2)
    labels = np.tile([0, 1], n_rows // 2)
    rng = np.random.RandomState(12)
    latent = 1.2 * labels + rng.normal(scale=0.8, size=n_rows)
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    reference = pd.DataFrame(
        {
            "source_row": source,
            "event_number": events,
            "y_true_light0_b1": labels,
            "self_configuring_por_probability_b": sigmoid(latent),
            "GN2v01_paper_Db_audit_only": latent + 0.2 * labels,
            "DL1dv01_paper_Db_audit_only": latent - 0.1 * labels,
        }
    )
    reference.to_csv(
        selfconfig / "self_configuring_locked_test_predictions.csv.gz",
        index=False,
    )
    np.savez_compressed(
        selfconfig / "frozen_split_indices.npz",
        sampled_source_rows=np.arange(2 * n_rows, dtype=np.int64),
        sampled_event_numbers=np.repeat(np.arange(n_rows, dtype=np.int64), 2),
        base=np.arange(20),
        operator_validation=np.arange(20, 40),
        derivative_gate=np.arange(40, 50),
        wp_validation=np.arange(50, 60),
        independent_test=np.arange(60, 120),
    )
    (selfconfig / "selected_derivative_gated_dimension.json").write_text(json.dumps({"K_DG": 3}))
    orbitals = reference[["source_row", "event_number", "y_true_light0_b1"]].copy()
    orbitals["phi_1"] = latent
    orbitals["phi_2"] = rng.normal(size=n_rows) + 0.4 * labels
    orbitals["phi_3"] = rng.normal(size=n_rows) - 0.3 * labels
    orbitals = orbitals.sample(frac=1.0, random_state=7)
    orbitals.to_csv(
        selfconfig / "self_configuring_locked_test_orbitals.csv.gz",
        index=False,
    )

    particle_prediction = reference[
        ["source_row", "event_number", "y_true_light0_b1"]
    ].copy()
    particle_prediction["particle_net_probability_b"] = sigmoid(
        latent + 0.25 * labels + rng.normal(scale=0.12, size=n_rows)
    )
    particle_prediction.to_csv(
        particle / "particle_net_locked_test_predictions.csv.gz",
        index=False,
    )
    manifest = {
        "method": "ParticleNet-JetSet-retrained",
        "architecture": "full",
        "model": {"paper_dense_units": 256},
        "source_h5_file": str(root / "tiny.h5"),
    }
    (particle / "particle_net_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_batch_loss": [0.5, 0.4, 0.3],
            "validation_auc": [0.7, 0.75, 0.8],
            "validation_log_loss": [0.5, 0.45, 0.4],
        }
    ).to_csv(particle / "particle_net_training_history.csv", index=False)

    n_rows *= 2
    labels = np.tile(labels, 2)
    events = np.repeat(np.arange(n_rows // 2, dtype=np.int64), 2)
    jet_dtype = np.dtype(
        [
            (LABEL_FIELD, "i4"),
            ("eventNumber", "i8"),
            ("pt_btagJes", "f4"),
            ("eta_btagJes", "f4"),
        ]
    )
    track_dtype = np.dtype(
        [("valid", "?")] + [(name, "f4") for name in TRACK_FIELDS]
    )
    jets = np.zeros(n_rows, dtype=jet_dtype)
    jets[LABEL_FIELD] = np.where(labels == 1, 5, 0)
    jets["eventNumber"] = events
    jets["pt_btagJes"] = rng.uniform(25_000, 200_000, size=n_rows)
    jets["eta_btagJes"] = rng.uniform(-2.4, 2.4, size=n_rows)
    tracks = np.zeros((n_rows, 40), dtype=track_dtype)
    tracks["valid"][:, :8] = True
    for index, name in enumerate(TRACK_FIELDS):
        tracks[name][:, :8] = rng.normal(
            loc=0.05 * index + 0.15 * labels[:, None],
            scale=1.0,
            size=(n_rows, 8),
        )
    tracks["numberOfPixelHits"][:, :8] = rng.randint(1, 6, size=(n_rows, 8))
    tracks["numberOfSCTHits"][:, :8] = rng.randint(2, 10, size=(n_rows, 8))
    h5_path = root / "tiny.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("jets", data=jets, chunks=(16,))
        handle.create_dataset("tracks", data=tracks, chunks=(16, 40))
    return selfconfig, particle, h5_path, reference


def test_strict_comparison_and_orbital_alignment(tmp_path):
    selfconfig, particle, _, reference = _make_fixture(tmp_path)
    loaded = load_comparison_inputs(selfconfig, particle)
    assert loaded.dimensions[POR_METHOD] == 3
    assert loaded.dimensions[PARTICLENET_METHOD] == 256
    path = selfconfig / "self_configuring_locked_test_orbitals.csv.gz"
    aligned = load_aligned_orbitals(path, reference)
    assert aligned["source_row"].tolist() == reference["source_row"].tolist()
    assert aligned[["orbital_1", "orbital_2", "orbital_3"]].notna().all().all()


def test_complete_result_workspace_run(tmp_path):
    selfconfig, particle, h5_path, _ = _make_fixture(tmp_path)
    output = tmp_path / "result"
    script = Path(__file__).resolve().parents[1] / "produce_paper_results.py"
    command = [
        sys.executable,
        str(script),
        str(selfconfig),
        "--particle-net-output",
        str(particle),
        "--h5-file",
        str(h5_path),
        "--bootstrap-repetitions",
        "25",
        "--require-orbital-analysis",
        "--allow-nonstandard-sample",
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = (
        "paper/TABLE_1_MATCHED_COMPARISON.tex",
        "paper/FIGURE_1_PERFORMANCE_AND_DIMENSION.pdf",
        "paper/FIGURE_1_PERFORMANCE_AND_DIMENSION.png",
        "paper/FIGURE_2_ORBITAL_CONTENT.pdf",
        "paper/RESULTS_TEXT.tex",
        "paper/PAPER_NUMBERS.tex",
        "audit/TABLE_AUDIT_GN2_DL1.tex",
        "statistics/paired_auc_differences_vs_dg_npor.csv",
        "diagnostics/alignment_report.json",
        "RESULTS_MANIFEST.json",
    )
    for relative in expected:
        assert (output / relative).exists(), relative
        assert (output / relative).stat().st_size > 0
    table = pd.read_csv(output / "paper/TABLE_1_MATCHED_COMPARISON.csv")
    assert table["method"].tolist() == [POR_METHOD, PARTICLENET_METHOD]
    assert table["decision_dimension"].tolist() == [3, 256]
    manifest = json.loads((output / "RESULTS_MANIFEST.json").read_text())
    assert manifest["best_por_only"]
    assert manifest["orbital_analysis"]["completed"]
