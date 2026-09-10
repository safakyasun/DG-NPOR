#!/usr/bin/env python3
"""JetSet baselines; no comparison probability enters the DG-NPOR score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import save_json
from por_hep.atlas_jetset import load_atlas_jetset_b_vs_light, verify_official_file
from por_hep.flavour_metrics import (
    background_rejection_at_signal_efficiency,
    event_cluster_replicate_indices,
    evaluate_flavour_tagging,
    roc_table,
)
from run_atlas_jetset_por import workspace_source_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare full DG-NPOR with leakage-safe diagnostic baselines."
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--h5-file",
        default="data/atlas_jetset/mc-flavtag-ttbar-small.h5",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _paired_auc_differences(y, groups, predictions, repetitions, random_state):
    rng = np.random.RandomState(int(random_state))
    reference = "DG-NPOR-full-PSD"
    samples = {name: [] for name in predictions if name != reference}
    completed = 0
    attempts = 0
    while completed < int(repetitions):
        attempts += 1
        if attempts > max(100, 10 * int(repetitions)):
            raise RuntimeError("Could not form two-class event bootstrap replicates.")
        chosen = event_cluster_replicate_indices(groups, rng)
        if len(np.unique(y[chosen])) != 2:
            continue
        reference_auc = roc_auc_score(y[chosen], predictions[reference][chosen])
        for name, score in predictions.items():
            if name != reference:
                samples[name].append(
                    float(reference_auc - roc_auc_score(y[chosen], score[chosen]))
                )
        completed += 1
    return {
        name: {
            "comparison": "AUC(DG-NPOR-full-PSD) - AUC(%s)" % name,
            "mean_difference": float(np.mean(values)),
            "lower_95_percentile": float(np.quantile(values, 0.025)),
            "upper_95_percentile": float(np.quantile(values, 0.975)),
            "repetitions": int(repetitions),
            "paired_event_cluster_bootstrap": True,
        }
        for name, values in samples.items()
    }


PAPER_WORKING_POINTS = (0.65, 0.70, 0.77, 0.85, 0.90)


def _paper_working_point_rows(method, y_true, score, score_source):
    rows = []
    for efficiency in PAPER_WORKING_POINTS:
        point = background_rejection_at_signal_efficiency(
            y_true, score, efficiency, confidence_level=0.68
        )
        rows.append(
            {
                "method": str(method),
                "score_source": str(score_source),
                "b_is_signal_label": 1,
                "light_is_background_label": 0,
                **point,
            }
        )
    return rows


def _finite_ratio(numerator, denominator):
    if not (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and float(denominator) > 0
    ):
        return float("nan")
    return float(numerator / denominator)


def _paper_rejection_ratios(working_points):
    frame = pd.DataFrame(working_points)
    rows = []
    for efficiency in PAPER_WORKING_POINTS:
        selected = frame[np.isclose(frame["target_b_efficiency"], efficiency)]
        values = dict(zip(selected["method"], selected["light_rejection"]))
        por = values["DG-NPOR-full-PSD"]
        gn2 = values["ATLAS-GN2v01-paper-Db"]
        dl1 = values["ATLAS-DL1dv01-paper-Db"]
        rows.append(
            {
                "target_b_efficiency": efficiency,
                "GN2_over_DL1d_light_rejection": _finite_ratio(gn2, dl1),
                "POR_over_DL1d_light_rejection": _finite_ratio(por, dl1),
                "POR_over_GN2_light_rejection": _finite_ratio(por, gn2),
                "ratio_censored_by_zero_observed_mistags": bool(
                    not (np.isfinite(gn2) and np.isfinite(dl1))
                ),
            }
        )
    return rows


def _save_paper_rejection_comparison(y_true, scores, working_points, output):
    tables = []
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        constrained_layout=True,
    )
    for method, score in scores.items():
        curve = roc_table(y_true, score, confidence_level=0.68)
        curve.insert(0, "method", method)
        tables.append(curve)
        finite = np.isfinite(curve["light_rejection"]) & (
            curve["light_rejection"] > 0
        )
        ax.plot(
            curve.loc[finite, "b_efficiency_tpr"],
            curve.loc[finite, "light_rejection"],
            linewidth=1.8,
            label=method,
        )
        finite_band = (
            np.isfinite(curve["light_rejection_lower_68"])
            & np.isfinite(curve["light_rejection_upper_68"])
            & (curve["light_rejection_lower_68"] > 0)
            & (curve["light_rejection_upper_68"] > 0)
        )
        ax.fill_between(
            curve.loc[finite_band, "b_efficiency_tpr"],
            curve.loc[finite_band, "light_rejection_lower_68"],
            curve.loc[finite_band, "light_rejection_upper_68"],
            alpha=0.10,
        )
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(output / "paper_ttbar_rejection_curves.csv", index=False)
    ratios = pd.DataFrame(_paper_rejection_ratios(working_points))
    ratios.to_csv(output / "paper_ttbar_rejection_ratios.csv", index=False)
    for column, label in (
        ("GN2_over_DL1d_light_rejection", "GN2 / DL1d"),
        ("POR_over_DL1d_light_rejection", "POR / DL1d"),
    ):
        finite = np.isfinite(ratios[column])
        ratio_ax.plot(
            ratios.loc[finite, "target_b_efficiency"],
            ratios.loc[finite, column],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    ax.set_xlim(0.55, 1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Light-jet rejection 1 / light efficiency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.01,
        0.02,
        "Shading: binomial 68% interval; zero-mistag points are censored",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    ratio_ax.axhline(1.0, color="black", linewidth=0.8)
    ratio_ax.set_xlabel("b-jet efficiency")
    ratio_ax.set_ylabel("Rejection ratio")
    ratio_ax.grid(True, alpha=0.25)
    ratio_handles, ratio_labels = ratio_ax.get_legend_handles_labels()
    if ratio_handles:
        ratio_ax.legend(ratio_handles, ratio_labels, fontsize=7, ncol=2)
    fig.savefig(output / "paper_ttbar_rejection_comparison.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    output = _resolve(args.output_dir)
    h5_file = _resolve(args.h5_file)
    predictions_path = output / "test_predictions.csv.gz"
    if not predictions_path.exists():
        raise FileNotFoundError("Score the locked DG-NPOR test before comparisons.")
    bundle = joblib.load(output / "trained_atlas_jetset_dg_npor_v9.joblib")
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

    splits = np.load(output / "frozen_split_indices.npz")
    development = np.concatenate(
        [splits["base"], splits["operator_validation"], splits["dimension_gate"]]
    )
    test = np.asarray(splits["independent_test"], dtype=int)
    y_test = dataset.y[test]

    stored = pd.read_csv(predictions_path)
    if not np.array_equal(
        stored["source_row"].to_numpy(dtype=int), dataset.row_index[test]
    ):
        raise RuntimeError("Stored DG-NPOR predictions do not match the locked test rows.")
    por_probability = stored["por_probability_b"].to_numpy(dtype=float)

    if bundle.get("analysis_protocol") == "paper-ttbar":
        paper_scores = {
            "DG-NPOR-full-PSD": por_probability,
            "ATLAS-GN2v01-paper-Db": dataset.audit_scores["GN2v01_paper_Db"][test],
            "ATLAS-DL1dv01-paper-Db": dataset.audit_scores["DL1dv01_paper_Db"][test],
        }
        paper_metrics = []
        for name, score in paper_scores.items():
            paper_metrics.extend(
                _paper_working_point_rows(
                    name,
                    y_test,
                    score,
                    (
                        "structured_POR_probability"
                        if name == "DG-NPOR-full-PSD"
                        else "paper_discriminant_audit_only"
                    ),
                )
            )
        pd.DataFrame(paper_metrics).to_csv(
            output / "paper_ttbar_working_point_comparison.csv", index=False
        )
        _save_paper_rejection_comparison(
            y_test, paper_scores, paper_metrics, output
        )
        save_json(
            output / "paper_ttbar_comparison_audit.json",
            {
                "protocol": "paper-ttbar b-versus-light slice",
                "primary_metric": "light-jet rejection versus b-jet efficiency",
                "central_operating_point_b_efficiency": 0.70,
                "working_points": list(PAPER_WORKING_POINTS),
                "confidence_intervals": "two-sided Clopper-Pearson 68% binomial intervals",
                "AUC_or_average_precision_used_for_paper_comparison": False,
                "fiducial_selection": dataset.schema_audit["protocol_audit"][
                    "physics_cuts"
                ],
                "GN2_Db": "log(pb / (0.20*pc + 0.05*ptau + 0.75*pu))",
                "DL1d_Db": "log(pb / (0.018*pc + 0.982*pu))",
                "same_locked_test_rows_for_all_methods": True,
                "GN2_DL1_scores_used_by_POR": False,
                "comparison_scope": "binary b-versus-light slice; not full four-class GN2 reproduction",
            },
        )
        central = pd.DataFrame(paper_metrics)
        central = central[np.isclose(central["target_b_efficiency"], 0.70)]
        print("Paper-matched ttbar comparison at the central 70% b-efficiency OP:")
        print(central.to_string(index=False))
        print("\nNo AUC or average-precision comparison was produced for paper-ttbar.")
        return 0

    # Generic diagnostic baselines need dense adapted development arrays.  Keep
    # this allocation below the paper-ttbar early return: the paper comparison
    # uses only stored POR predictions and audit-only GN2/DL1 discriminants.
    adapter = bundle["input_feature_adapter"]
    X_development = adapter.transform(dataset.X[development])
    X_test = adapter.transform(dataset.X[test])
    y_development = dataset.y[development]

    logistic = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    logistic.fit(X_development, y_development)
    logistic_probability = logistic.predict_proba(X_test)[:, 1]

    K = int(bundle["neural_por_model"].selected_k_)
    pca = PCA(n_components=K, svd_solver="randomized", random_state=42)
    X_development_pca = pca.fit_transform(X_development)
    pca_logistic = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    pca_logistic.fit(X_development_pca, y_development)
    pca_probability = pca_logistic.predict_proba(pca.transform(X_test))[:, 1]

    neural_probability = bundle[
        "neural_por_model"
    ].neural_geometry_.network_.predict_proba(X_test)[:, 1]
    predictions = {
        "DG-NPOR-full-PSD": por_probability,
        "adapted-137-logistic": logistic_probability,
        "PCA-KDG-logistic": pca_probability,
        "neural-auxiliary-sigmoid": neural_probability,
        "ATLAS-GN2v01-audit-score": dataset.audit_scores["GN2v01_b_vs_light"][test],
        "ATLAS-DL1dv01-audit-score": dataset.audit_scores["DL1dv01_b_vs_light"][test],
    }
    metrics = []
    for name, probability in predictions.items():
        metrics.append(
            evaluate_flavour_tagging(
                name,
                y_test,
                probability,
                "comparison_only_never_fused_into_DG_NPOR",
            )
        )
    pd.DataFrame(metrics).to_csv(output / "locked_test_baseline_comparison.csv", index=False)
    save_json(
        output / "paired_auc_difference_bootstrap.json",
        _paired_auc_differences(
            y_test,
            dataset.group_id[test],
            predictions,
            repetitions=args.bootstrap_repetitions,
            random_state=int(bundle["random_state"]) + 9001,
        ),
    )
    joblib.dump(
        {
            "adapted_137_logistic": logistic,
            "pca_KDG": pca,
            "pca_KDG_logistic": pca_logistic,
            "K_DG": K,
            "note": "comparison-only; GN2/DL1/PCA/logistic/neural probabilities never enter DG-NPOR",
        },
        output / "comparison_baselines.joblib",
        compress=3,
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
