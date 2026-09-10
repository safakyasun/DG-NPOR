#!/usr/bin/env python3
"""Evaluate a frozen, resolved JetSet DG-NPOR model on its locked test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import save_json
from por_hep.atlas_jetset import load_atlas_jetset_b_vs_light, verify_official_file
from run_atlas_jetset_por import _score_test, workspace_source_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Open the locked test exactly once for a frozen resolved run."
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--h5-file",
        default="data/atlas_jetset/mc-flavtag-ttbar-small.h5",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None):
    args = parse_args(argv)
    output = _resolve(args.output_dir)
    h5_file = _resolve(args.h5_file)
    bundle_path = output / "trained_atlas_jetset_dg_npor_v9.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Frozen model bundle not found: %s" % bundle_path)
    status_path = output / "test_scoring_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("scored"):
            raise RuntimeError("This locked test has already been scored.")

    bundle = joblib.load(bundle_path)
    if not bool(bundle["method_resolved_before_test"]):
        raise RuntimeError("Frozen method was unresolved; locked test must remain closed.")
    integrity = verify_official_file(h5_file, tier=bundle["dataset_tier"])
    if h5_file.name != bundle["official_file_name"]:
        raise RuntimeError("JetSet filename differs from the frozen run.")
    if int(integrity["size_bytes"]) != int(bundle["official_file_size"]):
        raise RuntimeError("JetSet file size differs from the frozen run.")
    if integrity["adler32"] != bundle["official_file_adler32"]:
        raise RuntimeError("JetSet Adler-32 differs from the frozen run.")
    dataset = load_atlas_jetset_b_vs_light(
        h5_file,
        sample=int(bundle["sample"]),
        random_state=int(bundle["random_state"]),
        label_definition=bundle["label_definition"],
        protocol=bundle.get("analysis_protocol", "pilot-b-light"),
        max_source_events=int(bundle.get("max_source_events", 2000000)),
    )
    if not np.array_equal(dataset.row_index, bundle["sampled_source_rows"]):
        raise RuntimeError("Sampled source rows differ from the frozen run.")
    if not np.array_equal(dataset.group_id, bundle["sampled_event_numbers"]):
        raise RuntimeError("Sampled event numbers differ from the frozen run.")
    if workspace_source_sha256() != bundle["source_tree_sha256"]:
        raise RuntimeError("Workspace source SHA256 differs from the frozen run.")

    test = np.asarray(bundle["independent_test_indices"], dtype=int)
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    repetitions = (
        int(args.bootstrap_repetitions)
        if args.bootstrap_repetitions is not None
        else int(run_config["bootstrap_repetitions"])
    )
    scoring_args = SimpleNamespace(
        query_batch_size=int(args.query_batch_size),
        random_state=int(bundle["random_state"]),
    )
    metrics = _score_test(
        bundle["neural_por_model"],
        bundle["input_feature_adapter"],
        dataset.X[test],
        dataset.y[test],
        dataset.row_index[test],
        dataset.group_id[test],
        dataset.audit_variables["jet_pt_btagJes_MeV"][test],
        dataset.audit_variables["jet_eta_btagJes"][test],
        output,
        {"bootstrap_repetitions": repetitions},
        scoring_args,
    )
    save_json(
        status_path,
        {
            "scored": True,
            "reason": "frozen resolved method evaluated by score_locked_test.py",
            "official_file_size_and_adler32_verified": True,
            "sampled_source_rows_verified": True,
            "sampled_event_numbers_verified": True,
            "source_tree_sha256_verified": True,
        },
    )
    print("Locked-test scoring completed")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
