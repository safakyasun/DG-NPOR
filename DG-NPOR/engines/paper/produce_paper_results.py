#!/usr/bin/env python3
"""Produce every table, figure, and number used by the paper Results section."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jetset_litcomp.metrics import (
    classification_summary,
    event_cluster_auc_bootstrap,
    roc_points,
    working_point,
)
from jetset_litcomp.orbital_analysis import (
    analyze_orbitals,
    reconstructed_feature_summaries,
)
from jetset_litcomp.paper_inputs import (
    DL1_METHOD,
    GN2_METHOD,
    PARTICLENET_METHOD,
    POR_METHOD,
    discover_orbital_file,
    load_aligned_orbitals,
    load_comparison_inputs,
)
from jetset_litcomp.paper_plots import (
    save_all_score_roc,
    save_orbital_content_figure,
    save_particle_training_figure,
    save_performance_dimension_figure,
)
from jetset_litcomp.paper_report import (
    audit_table_latex,
    main_table_latex,
    make_audit_table,
    make_main_table,
    paper_macros,
    results_text,
    write_text,
)


WORKSPACE_VERSION = "2.0.0"
EXPECTED_LOCKED_ROWS = 50115
TARGET_EFFICIENCIES = (0.60, 0.70, 0.77, 0.85)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication tables, figures, statistical files, and LaTeX "
            "snippets from frozen DG NPOR and ParticleNet predictions."
        )
    )
    parser.add_argument("selfconfig_output_dir")
    parser.add_argument("--particle-net-output", required=True)
    parser.add_argument("--h5-file")
    parser.add_argument("--orbital-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=2407)
    parser.add_argument("--orbital-batch-size", type=int, default=2048)
    parser.add_argument("--require-orbital-analysis", action="store_true")
    parser.add_argument("--allow-nonstandard-sample", action="store_true")
    return parser.parse_args()


def _safe_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _resolve_h5(explicit, particle_manifest):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    manifest_path = particle_manifest.get("source_h5_file")
    if manifest_path:
        candidates.append(Path(manifest_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if explicit:
        raise FileNotFoundError(f"ATLAS JetSet HDF5 file not found: {explicit}")
    return None


def _method_metric_table(y_true, scores, probability_methods, bootstrap_summary):
    rows = []
    for method, score in scores.items():
        row = classification_summary(
            y_true,
            score,
            probability=method in probability_methods,
        )
        row["method"] = method
        row["score_semantics"] = (
            "bottom probability" if method in probability_methods else "audit discriminant"
        )
        rows.append(row)
    metric = pd.DataFrame(rows)
    intervals = bootstrap_summary.drop(columns=["auc"]).copy()
    return metric.merge(intervals, on="method", how="left", validate="one_to_one")


def _working_points(y_true, scores):
    rows = []
    for method, score in scores.items():
        for efficiency in TARGET_EFFICIENCIES:
            row = working_point(y_true, score, efficiency, confidence=0.68)
            row["method"] = method
            rows.append(row)
    columns = ["method"] + [key for key in rows[0] if key != "method"]
    return pd.DataFrame(rows)[columns]


def _input_checksums(selfconfig_output, particle_output, orbital_path, h5_path):
    paths = {
        "dg_npor_locked_predictions": Path(selfconfig_output)
        / "self_configuring_locked_test_predictions.csv.gz",
        "frozen_split": Path(selfconfig_output) / "frozen_split_indices.npz",
        "particle_net_locked_predictions": Path(particle_output)
        / "particle_net_locked_test_predictions.csv.gz",
        "particle_net_manifest": Path(particle_output) / "particle_net_manifest.json",
    }
    if orbital_path:
        paths["selected_orbitals"] = Path(orbital_path)
    if h5_path:
        paths["atlas_jetset_h5"] = Path(h5_path)
    rows = []
    for role, path in paths.items():
        if path.exists():
            rows.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    paper_dir = output / "paper"
    audit_dir = output / "audit"
    statistics_dir = output / "statistics"
    diagnostics_dir = output / "diagnostics"
    curves_dir = output / "curves"
    for directory in (paper_dir, audit_dir, statistics_dir, diagnostics_dir, curves_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loaded = load_comparison_inputs(
        args.selfconfig_output_dir,
        args.particle_net_output,
    )
    reference = loaded.reference
    y_true = reference["y_true_light0_b1"].to_numpy(int)
    event_numbers = reference["event_number"].to_numpy(np.int64)
    # load_comparison_inputs verifies exact membership against frozen_split_indices,
    # which is stronger than one hard-coded row count and supports every profile.
    all_scores = {**loaded.main_scores, **loaded.audit_scores}
    print(
        "Locked test: %d jets, %d bottom, %d light, %d events"
        % (
            len(y_true),
            int(np.sum(y_true == 1)),
            int(np.sum(y_true == 0)),
            len(np.unique(event_numbers)),
        )
    )
    print(
        "Computing %d paired whole event bootstrap repetitions"
        % args.bootstrap_repetitions
    )
    bootstrap = event_cluster_auc_bootstrap(
        y_true,
        all_scores,
        event_numbers,
        repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
        reference_method=POR_METHOD,
    )
    bootstrap.summary.to_csv(statistics_dir / "auc_event_bootstrap_intervals.csv", index=False)
    bootstrap.paired_differences.to_csv(
        statistics_dir / "paired_auc_differences_vs_dg_npor.csv",
        index=False,
    )
    bootstrap.replicate_values.to_csv(
        statistics_dir / "auc_event_bootstrap_replicates.csv.gz",
        index=False,
    )

    metrics = _method_metric_table(
        y_true,
        all_scores,
        loaded.probability_methods,
        bootstrap.summary,
    )
    metrics.to_csv(statistics_dir / "all_locked_test_metrics.csv", index=False)
    working_points = _working_points(y_true, all_scores)
    working_points.to_csv(
        statistics_dir / "working_points_60_70_77_85.csv",
        index=False,
    )
    for method, score in all_scores.items():
        roc_points(y_true, score).to_csv(
            curves_dir / f"roc_{_safe_name(method)}.csv.gz",
            index=False,
        )

    main_table = make_main_table(metrics, working_points, loaded.dimensions)
    main_table.to_csv(paper_dir / "TABLE_1_MATCHED_COMPARISON.csv", index=False)
    write_text(
        paper_dir / "TABLE_1_MATCHED_COMPARISON.tex",
        main_table_latex(main_table),
    )
    audit_table = make_audit_table(
        metrics,
        working_points,
        (GN2_METHOD, DL1_METHOD),
    )
    audit_table.to_csv(audit_dir / "TABLE_AUDIT_GN2_DL1.csv", index=False)
    write_text(
        audit_dir / "TABLE_AUDIT_GN2_DL1.tex",
        audit_table_latex(audit_table),
    )

    locked_events = int(len(np.unique(event_numbers)))
    write_text(
        paper_dir / "PAPER_NUMBERS.tex",
        paper_macros(main_table, len(reference), locked_events),
    )
    write_text(
        paper_dir / "RESULTS_TEXT.tex",
        results_text(
            main_table,
            bootstrap.paired_differences,
            len(reference),
            locked_events,
        ),
    )
    main_table.to_json(
        paper_dir / "PAPER_NUMBERS.json",
        orient="records",
        indent=2,
    )

    save_performance_dimension_figure(
        y_true,
        loaded.main_scores,
        main_table,
        paper_dir / "FIGURE_1_PERFORMANCE_AND_DIMENSION",
    )
    save_all_score_roc(
        y_true,
        loaded.main_scores,
        loaded.audit_scores,
        metrics,
        audit_dir / "FIGURE_AUDIT_COMMON_TEST_ROC",
    )
    training_figure_written = save_particle_training_figure(
        Path(args.particle_net_output) / "particle_net_training_history.csv",
        diagnostics_dir / "FIGURE_PARTICLENET_TRAINING_HISTORY",
    )

    orbital_path = discover_orbital_file(
        args.selfconfig_output_dir,
        explicit=args.orbital_file,
    )
    h5_path = _resolve_h5(args.h5_file, loaded.particle_net_manifest)
    orbital_status = {
        "requested": bool(args.require_orbital_analysis),
        "orbital_file": str(orbital_path.resolve()) if orbital_path else None,
        "h5_file": str(h5_path) if h5_path else None,
        "completed": False,
        "reason": None,
    }
    if orbital_path is not None and h5_path is not None:
        print("Computing descriptive content analysis for the selected orbitals")
        orbitals = load_aligned_orbitals(orbital_path, reference)
        from jetset_litcomp.paper_inputs import identify_orbital_columns
        if len(identify_orbital_columns(orbitals.columns)) != loaded.dimensions[POR_METHOD]:
            raise ValueError("Exported orbital dimension differs from frozen selection metadata.")
        features = reconstructed_feature_summaries(
            h5_path,
            reference,
            batch_size=args.orbital_batch_size,
            max_tracks=int(loaded.particle_net_manifest.get("inputs", {}).get("max_tracks", 20)),
        )
        analysis = analyze_orbitals(orbitals, features)
        analysis["class_summary"].to_csv(
            statistics_dir / "orbital_class_summaries.csv",
            index=False,
        )
        analysis["feature_correlations"].to_csv(
            statistics_dir / "orbital_feature_spearman.csv",
            index=False,
        )
        analysis["group_representatives"].to_csv(
            paper_dir / "TABLE_2_ORBITAL_GROUP_ASSOCIATIONS.csv",
            index=False,
        )
        features.to_csv(
            diagnostics_dir / "locked_test_reconstructed_feature_summaries.csv.gz",
            index=False,
        )
        save_orbital_content_figure(
            analysis,
            paper_dir / "FIGURE_2_ORBITAL_CONTENT",
        )
        orbital_status["completed"] = True
    else:
        missing = []
        if orbital_path is None:
            missing.append("a locked test export containing all selected orbitals")
        if h5_path is None:
            missing.append("the ATLAS JetSet HDF5 path")
        orbital_status["reason"] = "Missing " + " and ".join(missing)
        if args.require_orbital_analysis:
            raise RuntimeError(orbital_status["reason"])
    _write_json(diagnostics_dir / "orbital_analysis_status.json", orbital_status)

    checksums = _input_checksums(
        args.selfconfig_output_dir,
        args.particle_net_output,
        orbital_path,
        h5_path,
    )
    checksums.to_csv(diagnostics_dir / "input_checksums.csv", index=False)
    alignment = {
        "locked_test_rows": int(len(reference)),
        "locked_test_bottom_jets": int(np.sum(y_true == 1)),
        "locked_test_light_jets": int(np.sum(y_true == 0)),
        "locked_test_unique_events": locked_events,
        "alignment_keys": ["source_row", "event_number", "y_true_light0_b1"],
        "particle_net_rows_exactly_matched": True,
        "duplicated_alignment_keys": False,
        "event_disjoint_split_source": "frozen_split_indices.npz",
    }
    _write_json(diagnostics_dir / "alignment_report.json", alignment)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    _write_json(diagnostics_dir / "software_environment.json", environment)
    manifest = {
        "workspace_version": WORKSPACE_VERSION,
        "comparison_population": "same ATLAS JetSet locked test jets",
        "main_table_methods": [POR_METHOD, PARTICLENET_METHOD],
        "audit_only_methods": [GN2_METHOD, DL1_METHOD],
        "best_por_only": True,
        "historical_por_ablations_in_main_results": False,
        "decision_dimensions": loaded.dimensions,
        "primary_metric": "light rejection at 70 percent bottom jet efficiency",
        "secondary_working_points": list(TARGET_EFFICIENCIES),
        "auc_uncertainty": "paired whole event cluster bootstrap",
        "efficiency_uncertainty": "two sided Clopper Pearson 68 percent",
        "working_point_definition": "test resolved empirical bottom score quantile",
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "random_state": int(args.random_state),
        "orbital_analysis": orbital_status,
        "particle_net_training_figure_written": training_figure_written,
        "claim_boundary": (
            "GN2v01 and DL1dv01 are same jet audit scores, not matched input or "
            "matched training budget baselines."
        ),
    }
    _write_json(output / "RESULTS_MANIFEST.json", manifest)

    print("Paper outputs written to", paper_dir.resolve())
    print(main_table.to_string(index=False))
    if not orbital_status["completed"]:
        print("Orbital content figure not written:", orbital_status["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
