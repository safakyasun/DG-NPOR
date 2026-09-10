#!/usr/bin/env python3
"""Open the locked residue-9 test once for the frozen structured-track POR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import predict_batches, save_json
from por_hep.atlas_jetset import verify_official_file
from por_hep.flavour_metrics import background_rejection_at_signal_efficiency
from por_hep.structured_tracks import H5StructuredTrackSource, load_atlas_jetset_index
from run_atlas_jetset_por import workspace_source_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the frozen Set/Graph-POR model on its locked test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_dir")
    parser.add_argument("--h5-file", required=True)
    parser.add_argument("--read-batch-size", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None):
    args = parse_args(argv)
    output = _resolve(args.output_dir)
    h5_file = _resolve(args.h5_file)
    bundle_path = output / "trained_structured_track_dg_npor_v9.joblib"
    status_path = output / "test_scoring_status.json"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("scored"):
            raise RuntimeError("This locked test has already been scored.")
    bundle = joblib.load(bundle_path)
    if not bool(bundle["method_resolved_before_test"]):
        raise RuntimeError("Frozen structured method is unresolved.")
    integrity = verify_official_file(h5_file, tier=bundle["dataset_tier"])
    if h5_file.name != bundle["official_file_name"]:
        raise RuntimeError("JetSet filename differs from the frozen run.")
    if int(integrity["size_bytes"]) != int(bundle["official_file_size"]):
        raise RuntimeError("JetSet size differs from the frozen run.")
    if integrity["adler32"] != bundle["official_file_adler32"]:
        raise RuntimeError("JetSet Adler-32 differs from the frozen run.")
    if workspace_source_sha256() != bundle["source_tree_sha256"]:
        raise RuntimeError("Workspace source SHA256 differs from the frozen run.")

    dataset = load_atlas_jetset_index(
        h5_file,
        sample=int(bundle["sample"]),
        random_state=int(bundle["random_state"]),
        label_definition=bundle["label_definition"],
        protocol=bundle["analysis_protocol"],
        max_source_events=int(bundle["max_source_events"]),
    )
    if not np.array_equal(dataset.row_index, bundle["sampled_source_rows"]):
        raise RuntimeError("Sampled source rows differ from the frozen scan.")
    if not np.array_equal(dataset.group_id, bundle["sampled_event_numbers"]):
        raise RuntimeError("Sampled event numbers differ from the frozen scan.")
    test = np.asarray(bundle["independent_test_indices"], dtype=int)
    encoder = bundle["structured_track_encoder"]
    model = bundle["neural_por_model"]
    adapter = bundle["input_feature_adapter"]
    with H5StructuredTrackSource(
        h5_file,
        dataset.row_index,
        max_tracks=int(bundle["max_tracks"]),
    ) as source:
        encoded = encoder.encode_source(
            source, test, batch_size=args.read_batch_size
        )
    prediction = predict_batches(
        model,
        adapter,
        encoded["geometry"],
        args.query_batch_size,
    )
    probability = prediction["probability"]
    y_test = dataset.y[test]

    por_working_points = []
    for efficiency in (0.60, 0.70, 0.77, 0.85):
        row = background_rejection_at_signal_efficiency(
            y_test, probability, efficiency
        )
        row["method"] = "%s-POR-full-PSD" % bundle["selected_representation"].title()
        row["score_source"] = "structured_track_POR_probability"
        por_working_points.append(row)
    pd.DataFrame(por_working_points).to_csv(
        output / "structured_por_locked_test_working_points.csv", index=False
    )

    comparison = []
    for method, score, source_name in (
        (
            "%s-POR-full-PSD" % bundle["selected_representation"].title(),
            probability,
            "structured_track_POR_probability",
        ),
        (
            "ATLAS-GN2v01-paper-Db",
            dataset.audit_scores["GN2v01_paper_Db"][test],
            "paper_discriminant_audit_only",
        ),
        (
            "ATLAS-DL1dv01-paper-Db",
            dataset.audit_scores["DL1dv01_paper_Db"][test],
            "paper_discriminant_audit_only",
        ),
    ):
        row = background_rejection_at_signal_efficiency(y_test, score, 0.70)
        row["method"] = method
        row["score_source"] = source_name
        comparison.append(row)
    columns = ["method", "score_source"] + [
        name for name in comparison[0] if name not in {"method", "score_source"}
    ]
    comparison_frame = pd.DataFrame(comparison)[columns]
    comparison_frame.to_csv(
        output / "paper_matched_ttbar_locked_test_comparison.csv", index=False
    )
    pd.DataFrame(
        {
            "source_row": dataset.row_index[test],
            "event_number": dataset.group_id[test],
            "y_true_light0_b1": y_test,
            "structured_por_probability_b": probability,
            "diagonal_only_probability_b": prediction[
                "diagonal_only_probability"
            ],
            "structured_encoder_auxiliary_probability_audit_only": encoded[
                "auxiliary_probability"
            ],
            "GN2v01_paper_Db_audit_only": dataset.audit_scores[
                "GN2v01_paper_Db"
            ][test],
            "DL1dv01_paper_Db_audit_only": dataset.audit_scores[
                "DL1dv01_paper_Db"
            ][test],
        }
    ).to_csv(output / "structured_locked_test_predictions.csv.gz", index=False)
    save_json(
        status_path,
        {
            "scored": True,
            "selected_representation": bundle["selected_representation"],
            "official_file_size_and_adler32_verified": True,
            "sampled_source_rows_verified": True,
            "sampled_event_numbers_verified": True,
            "source_tree_sha256_verified": True,
            "AUC_or_average_precision_comparison_produced": False,
        },
    )
    print("Locked-test structured-track scoring completed")
    print("\nPaper-matched ttbar comparison at the central 70% b-efficiency OP:")
    print(comparison_frame.to_string(index=False))
    print("\nNo AUC or average-precision comparison was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
