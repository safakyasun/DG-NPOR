import h5py
import numpy as np
import subprocess

import por_hep.atlas_jetset as atlas_jetset

from por_hep.atlas_jetset import (
    AtlasJetSetFeatureMap,
    RECONSTRUCTED_JET_INPUT_FIELDS,
    RECONSTRUCTED_TRACK_INPUT_FIELDS,
    dl1d_paper_b_discriminant,
    gn2_paper_b_discriminant,
    jetset_feature_names,
    load_atlas_jetset_b_vs_light,
)
from run_atlas_jetset_por import _role_audit, _split_roles


def _mock_file(path, n=80):
    jet_fields = [
        ("pt_btagJes", "f4"),
        ("eta_btagJes", "f4"),
        ("HadronConeExclTruthLabelID", "i4"),
        ("HadronGhostTruthLabelID", "i4"),
        ("eventNumber", "i8"),
        ("GN2v01_pb", "f4"),
        ("GN2v01_pc", "f4"),
        ("GN2v01_pu", "f4"),
        ("GN2v01_ptau", "f4"),
        ("DL1dv01_pb", "f4"),
        ("DL1dv01_pc", "f4"),
        ("DL1dv01_pu", "f4"),
    ]
    track_fields = [(name, "f4") for name in RECONSTRUCTED_TRACK_INPUT_FIELDS]
    track_fields.append(("valid", "?"))
    jets = np.zeros(n, dtype=jet_fields)
    events = np.arange(n, dtype=np.int64)
    labels = np.where((events // 20) % 2 == 0, 0, 5)
    jets["pt_btagJes"] = 30000.0 + 100.0 * events
    jets["eta_btagJes"] = np.linspace(-2.4, 2.4, n)
    jets["HadronConeExclTruthLabelID"] = labels
    jets["HadronGhostTruthLabelID"] = labels
    jets["eventNumber"] = events
    jets["GN2v01_pb"] = np.where(labels == 5, 0.8, 0.2)
    jets["GN2v01_pc"] = 0.05
    jets["GN2v01_ptau"] = 0.02
    jets["GN2v01_pu"] = 1.0 - (
        jets["GN2v01_pb"] + jets["GN2v01_pc"] + jets["GN2v01_ptau"]
    )
    jets["DL1dv01_pb"] = np.where(labels == 5, 0.7, 0.3)
    jets["DL1dv01_pc"] = 0.1
    jets["DL1dv01_pu"] = 1.0 - (
        jets["DL1dv01_pb"] + jets["DL1dv01_pc"]
    )
    tracks = np.zeros((n, 40), dtype=track_fields)
    tracks["valid"][:, :6] = True
    for index, name in enumerate(RECONSTRUCTED_TRACK_INPUT_FIELDS):
        tracks[name][:, :6] = (index + 1) * 0.01 + np.arange(6) * 0.001
    with h5py.File(path, "w") as handle:
        handle.create_dataset("jets", data=jets, chunks=(16,), compression="gzip")
        handle.create_dataset("tracks", data=tracks, chunks=(16, 40), compression="gzip")
        handle.create_dataset("eventwise", data=np.zeros(n, dtype=[("nJets", "u8")]))
        handle.create_dataset("truth_hadrons", data=np.zeros((n, 5), dtype=[("valid", "?")]))


def test_loader_is_balanced_finite_and_leakage_safe(tmp_path):
    path = tmp_path / "mock.h5"
    _mock_file(path)
    dataset = load_atlas_jetset_b_vs_light(path, sample=40, random_state=42)
    assert dataset.X.shape == (40, 137)
    assert dataset.feature_names == jetset_feature_names()
    assert np.isfinite(dataset.X).all()
    assert np.bincount(dataset.y, minlength=2).tolist() == [20, 20]
    assert dataset.schema_audit["forbidden_input_overlap"] == []
    assert dataset.schema_audit["GN2_or_DL1_scores_used_as_input"] is False
    assert dataset.schema_audit["truth_hadrons_dataset_used_as_input"] is False
    assert set(RECONSTRUCTED_JET_INPUT_FIELDS) == {"pt_btagJes", "eta_btagJes"}
    adapter = AtlasJetSetFeatureMap(dataset.feature_names)
    transformed = adapter.fit_transform(dataset.X[:20])
    assert transformed.shape == (20, 137)
    assert np.isfinite(transformed).all()


def test_event_modulo_roles_have_no_event_overlap():
    events = np.arange(80, dtype=np.int64)
    y = ((events // 20) % 2).astype(int)
    rows = np.arange(80, dtype=int)
    roles = _split_roles(y, events)
    audit = _role_audit(roles, y, rows, events)
    assert audit["split_uses_labels"] is False
    assert audit["event_leakage_detected"] is False
    for pair in audit["cross_role_source_intersections"].values():
        assert pair == {"source_rows": 0, "event_numbers": 0}


def test_paper_ttbar_protocol_applies_fiducial_selection_and_discriminants(tmp_path):
    path = tmp_path / "mock.h5"
    _mock_file(path)
    dataset = load_atlas_jetset_b_vs_light(
        path,
        sample=40,
        random_state=42,
        protocol="paper-ttbar",
        max_source_events=2000000,
    )
    audit = dataset.schema_audit["protocol_audit"]
    assert audit["analysis_protocol"] == "paper-ttbar"
    assert audit["source_event_cap_binding"] is False
    assert dataset.schema_audit["official_eventwise_rows"] == 80
    assert np.all(dataset.audit_variables["jet_pt_btagJes_MeV"] > 20000)
    assert np.all(dataset.audit_variables["jet_pt_btagJes_MeV"] < 250000)
    assert np.all(np.abs(dataset.audit_variables["jet_eta_btagJes"]) < 2.5)
    assert np.isfinite(dataset.audit_scores["GN2v01_paper_Db"]).all()
    assert np.isfinite(dataset.audit_scores["DL1dv01_paper_Db"]).all()


def test_paper_discriminant_equations():
    gn2 = gn2_paper_b_discriminant(
        np.array([0.6]), np.array([0.1]), np.array([0.1]), np.array([0.2])
    )
    dl1 = dl1d_paper_b_discriminant(
        np.array([0.6]), np.array([0.1]), np.array([0.3])
    )
    assert np.allclose(gn2, np.log(0.6 / (0.20 * 0.1 + 0.05 * 0.1 + 0.75 * 0.2)))
    assert np.allclose(dl1, np.log(0.6 / (0.018 * 0.1 + 0.982 * 0.3)))


def test_download_resumes_preserved_partial_after_curl_18(tmp_path, monkeypatch):
    destination = tmp_path / "mc-flavtag-ttbar-small.h5"
    partial = tmp_path / "mc-flavtag-ttbar-small.h5.part"
    partial.write_bytes(b"already-downloaded")
    calls = []

    def fake_run(command, check):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(18, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(atlas_jetset.shutil, "which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr(atlas_jetset.subprocess, "run", fake_run)
    monkeypatch.setattr(atlas_jetset.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        atlas_jetset,
        "verify_official_file",
        lambda path, tier: {
            "tier": tier,
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "adler32": "test",
        },
    )

    result = atlas_jetset.download_official_file(destination, tier="small")

    assert len(calls) == 2
    assert "--continue-at" in calls[0]
    assert calls[0][calls[0].index("--continue-at") + 1] == "-"
    assert calls[0][-1] == str(partial)
    assert destination.read_bytes() == b"already-downloaded"
    assert result["resumed_from_bytes"] == len(b"already-downloaded")
