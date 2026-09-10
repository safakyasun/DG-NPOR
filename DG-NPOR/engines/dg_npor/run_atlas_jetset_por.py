#!/usr/bin/env python3
"""Unified V9 DG-NPOR for ATLAS JetSet b- versus light-jet tagging."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import (
    compactness_report,
    index_sha256,
    measurement_summary,
    predict_batches,
    save_json,
)
from por_core import DerivativeGatedNeuralPOR
from por_hep.atlas_jetset import (
    AtlasJetSetFeatureMap,
    JETSET_PROTOCOLS,
    download_official_file,
    infer_official_tier,
    load_atlas_jetset_b_vs_light,
    verify_official_file,
)
from por_hep.flavour_metrics import (
    event_cluster_bootstrap_intervals,
    evaluate_flavour_tagging,
    kinematic_slice_metrics,
    roc_table,
)


def preset_defaults(preset):
    if preset == "quick":
        return {
            "sample": 20000,
            "graph_sample": 6000,
            "k_max": 64,
            "dimension_bootstrap_repetitions": 50,
            "neural_max_epochs": 120,
            "neural_patience": 15,
            "measurement_sample": 4000,
            "measurement_max_iter": 800,
            "measurement_restarts": 1,
            "bootstrap_repetitions": 100,
            "lambda_pi_grid": (0.10, 0.25, 0.50, 1.0),
        }
    return {
        "sample": 100000,
        "graph_sample": 10000,
        "k_max": 128,
        "dimension_bootstrap_repetitions": 200,
        "neural_max_epochs": 250,
        "neural_patience": 25,
        "measurement_sample": 8000,
        "measurement_max_iter": 1200,
        "measurement_restarts": 2,
        "bootstrap_repetitions": 300,
        "lambda_pi_grid": (0.05, 0.10, 0.25, 0.50, 1.0, 2.0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "ATLAS JetSet b/light flavour tagging with h_theta geometry, "
            "H=L_sym+lambda_pi*V_PI, dJ/dgate orbitals, and structured PSD."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample", type=int, default=None, help="balanced b+light jets; 0 uses all available balanced jets")
    parser.add_argument(
        "--protocol",
        choices=JETSET_PROTOCOLS,
        default="pilot-b-light",
        help="paper-ttbar applies the GN2 low-pT ttbar fiducial selection",
    )
    parser.add_argument(
        "--max-source-events",
        type=int,
        default=2000000,
        help=(
            "label-blind analysis-pool event cap; this does not shrink the "
            "monolithic official HDF5 download"
        ),
    )
    parser.add_argument("--dataset-tier", choices=("small", "medium", "large"), default="small")
    parser.add_argument(
        "--h5-file",
        default="data/atlas_jetset/mc-flavtag-ttbar-small.h5",
        help="official ATLAS JetSet HDF5 path",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="download the selected official tier from CERN Open Data",
    )
    parser.add_argument("--label-definition", choices=("cone", "ghost"), default="cone")
    parser.add_argument("--output-dir", default="outputs/atlas_jetset_b_light_quick")
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--scale-neighbor", type=int, default=16)
    parser.add_argument("--graph-sample", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--dimension-bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--neural-max-epochs", type=int, default=None)
    parser.add_argument("--neural-patience", type=int, default=None)
    parser.add_argument("--measurement-sample", type=int, default=None)
    parser.add_argument("--measurement-max-iter", type=int, default=None)
    parser.add_argument("--measurement-restarts", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--defer-test-scoring",
        action="store_true",
        help="Freeze the model and test indices without evaluating the locked test.",
    )
    return parser.parse_args(argv)


def _resolved_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def workspace_source_sha256(project_root=PROJECT_ROOT):
    """Hash package, backend, and all JetSet entry-point source files."""
    digest = hashlib.sha256()
    paths = list((Path(project_root) / "src").rglob("*.py"))
    paths.extend((Path(project_root) / "examples").rglob("*.py"))
    paths.extend(Path(project_root).glob("*.py"))
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(project_root))):
        relative = str(path.relative_to(project_root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _settings(args):
    defaults = preset_defaults(args.preset)
    names = (
        "sample",
        "graph_sample",
        "k_max",
        "dimension_bootstrap_repetitions",
        "neural_max_epochs",
        "neural_patience",
        "measurement_sample",
        "measurement_max_iter",
        "measurement_restarts",
        "bootstrap_repetitions",
    )
    values = {
        name: getattr(args, name) if getattr(args, name) is not None else defaults[name]
        for name in names
    }
    values["lambda_pi_grid"] = defaults["lambda_pi_grid"]
    return values


def _split_roles(y, event_numbers):
    """Reproduce the public ATLAS eventNumber modulo-10 partition.

    ATLAS uses residues 0..7 for training, 8 for validation, and 9 for test.
    Unified V9 allocates 0..5 to the base geometry and 6..7 to operator
    validation while preserving the official validation and test residues.
    The assignment is label-blind and all jets from an event stay together.
    """
    event_numbers = np.asarray(event_numbers, dtype=np.int64)
    residue = np.mod(event_numbers, 10)
    roles = {
        "base": np.flatnonzero(residue <= 5).astype(int),
        "operator_validation": np.flatnonzero((residue == 6) | (residue == 7)).astype(int),
        "dimension_gate": np.flatnonzero(residue == 8).astype(int),
        "independent_test": np.flatnonzero(residue == 9).astype(int),
    }
    all_indices = np.concatenate(list(roles.values()))
    if len(np.unique(all_indices)) != len(y):
        raise RuntimeError("Split roles are not a disjoint partition.")
    for name, indices in roles.items():
        if len(indices) == 0 or len(np.unique(np.asarray(y)[indices])) != 2:
            raise RuntimeError("Role %s does not contain both classes." % name)
    return roles


def _role_audit(roles, y, source_rows, event_numbers):
    names = list(roles)
    intersections = {}
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            row_overlap = np.intersect1d(source_rows[roles[first]], source_rows[roles[second]])
            event_overlap = np.intersect1d(event_numbers[roles[first]], event_numbers[roles[second]])
            intersections["%s__%s" % (first, second)] = {
                "source_rows": int(len(row_overlap)),
                "event_numbers": int(len(event_overlap)),
            }
            if len(row_overlap) or len(event_overlap):
                raise RuntimeError("Source-row or event leakage between split roles.")
    return {
        "protocol": "ATLAS-public-eventNumber-mod10: base=0..5 operator=6..7 dimension=8 test=9",
        "split_uses_labels": False,
        "roles": {
            name: {
                "rows": int(len(indices)),
                "unique_events": int(len(np.unique(event_numbers[indices]))),
                "class_counts": {
                    str(int(label)): int(count)
                    for label, count in zip(*np.unique(y[indices], return_counts=True))
                },
                "source_index_sha256": index_sha256(source_rows[indices]),
            }
            for name, indices in roles.items()
        },
        "cross_role_source_intersections": intersections,
        "source_row_leakage_detected": False,
        "event_leakage_detected": False,
        "test_labels_used_for_selection": False,
    }


def _save_roc_plot(table, output_path):
    finite = np.isfinite(table["light_rejection"]) & (table["light_rejection"] > 0)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(
        table.loc[finite, "b_efficiency_tpr"],
        table.loc[finite, "light_rejection"],
        linewidth=2.0,
    )
    ax.set_yscale("log")
    ax.set_xlabel("b-jet efficiency")
    ax.set_ylabel("Light-jet rejection 1/FPR")
    ax.set_title("DG-NPOR V9 ATLAS JetSet flavour tagging")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_core_artifacts(model, base_source_rows, output, args):
    core = model.core_
    graph_positions = np.asarray(model.selection_training_indices_, dtype=int)
    graph_rows = np.asarray(base_source_rows, dtype=int)[graph_positions]
    pd.DataFrame(
        {
            "source_row": graph_rows,
            "truth_label_light0_b1": core.base.y,
            "posterior_light_y0": core.base.posterior[:, 0],
            "posterior_b_y1": core.base.posterior[:, 1],
            "J_predictive_information": core.base.information,
            "V_PI": core.base.potential,
            "graph_degree": core.base.graph.degree,
        }
    ).to_csv(output / "predictive_information_field.csv.gz", index=False)

    bank_width = int(model.candidate_bank_k_)
    orbitals = pd.DataFrame(
        core.eigensystem.phi[:, :bank_width],
        columns=["phi_%03d" % (index + 1) for index in range(bank_width)],
    )
    orbitals.insert(0, "source_row", graph_rows)
    orbitals.to_csv(output / "predictive_orbitals_landmarks.csv.gz", index=False)
    pd.DataFrame(
        {
            "orbital_one_based": np.arange(1, len(model.selection_eigenvalues_) + 1),
            "eigenvalue": model.selection_eigenvalues_,
            "inside_candidate_bank": np.arange(len(model.selection_eigenvalues_))
            < bank_width,
            "selected_by_derivative_gate": np.isin(
                np.arange(len(model.selection_eigenvalues_)),
                model.selected_orbital_indices_,
            ),
        }
    ).to_csv(output / "por_eigenvalues_and_selection.csv", index=False)
    sparse.save_npz(output / "knn_affinity_W.npz", core.base.graph.W)
    sparse.save_npz(output / "normalized_laplacian_L_sym.npz", core.eigensystem.L_sym)
    sparse.save_npz(output / "por_operator_H.npz", core.eigensystem.H_N)
    save_json(
        output / "graph_and_operator_summary.json",
        {
            "graph": "sparse self-tuning symmetric kNN in learned h_theta geometry",
            "landmark_rows": int(len(core.base.X)),
            "n_neighbors": int(args.neighbors),
            "scale_neighbor": int(args.scale_neighbor),
            "affinity_nonzeros": int(core.base.graph.W.nnz),
            "degree_min": float(np.min(core.base.graph.degree)),
            "degree_median": float(np.median(core.base.graph.degree)),
            "degree_max": float(np.max(core.base.graph.degree)),
            "global_class_prior_light_b": core.base.priors.tolist(),
            "predictive_information": "J=KL(P(Y|x)||P(Y))",
            "potential": "V_PI=B_pi-J",
            "operator": "H=L_sym+lambda_pi*V_PI",
            "lambda_pi": float(model.selected_operator_parameters_),
        },
    )


def _save_model_paths(model, output):
    pd.DataFrame(model.neural_training_path()).to_csv(
        output / "neural_geometry_training.csv", index=False
    )
    pd.DataFrame(model.operator_selection_records_).to_csv(
        output / "operator_selection_path.csv", index=False
    )
    pd.DataFrame(model.candidate_bank_checkpoint_path()).to_csv(
        output / "candidate_bank_spectrum_checkpoints.csv", index=False
    )
    pd.DataFrame(model.candidate_bank_path()).to_csv(
        output / "candidate_predictive_field_spectrum.csv", index=False
    )
    pd.DataFrame(model.dimension_checkpoint_path()).to_csv(
        output / "derivative_gate_candidate_bank.csv", index=False
    )
    pd.DataFrame(model.dimension_path()).to_csv(
        output / "derivative_orbital_importance.csv", index=False
    )
    pd.DataFrame(model.risk_audit_path()).to_csv(
        output / "predictive_risk_audit.csv", index=False
    )


def _score_test(
    model,
    adapter,
    X_test,
    y_test,
    source_rows,
    event_numbers,
    pt_mev,
    eta,
    output,
    settings,
    args,
):
    prediction = predict_batches(model, adapter, X_test, args.query_batch_size)
    metrics = evaluate_flavour_tagging(
        "ATLAS-JetSet-b-light-DG-NPOR-V9",
        y_test,
        prediction["probability"],
        "dJ_dgate_sparse_por_psd_no_fusion_no_calibration",
    )
    diagonal = evaluate_flavour_tagging(
        "ATLAS-JetSet-b-light-DG-NPOR-V9-diagonal-only",
        y_test,
        prediction["diagonal_only_probability"],
        "same_state_diagonal_only_psd_ablation",
    )
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([metrics, diagonal]).to_csv(output / "ablation_metrics.csv", index=False)
    save_json(output / "jet_tagging_metrics.json", metrics)
    save_json(
        output / "test_confidence_intervals.json",
        event_cluster_bootstrap_intervals(
            y_test,
            prediction["probability"],
            event_numbers,
            repetitions=settings["bootstrap_repetitions"],
            random_state=args.random_state + 5001,
        ),
    )
    predictions = pd.DataFrame(
        {
            "source_row": source_rows,
            "event_number": event_numbers,
            "y_true_light0_b1": y_test,
            "por_probability_b": prediction["probability"],
            "diagonal_only_probability_b": prediction[
                "diagonal_only_probability"
            ],
            "diagonal_b_contribution": prediction[
                "diagonal_signal_contribution"
            ],
            "cross_orbital_b_contribution": prediction[
                "cross_orbital_signal_contribution"
            ],
            "jet_pt_btagJes_MeV": pt_mev,
            "jet_eta_btagJes": eta,
        }
    )
    predictions.to_csv(output / "test_predictions.csv.gz", index=False)
    roc = roc_table(y_test, prediction["probability"])
    roc.to_csv(output / "roc_curve.csv", index=False)
    _save_roc_plot(roc, output / "roc_curve.png")
    kinematic_slice_metrics(
        y_test, prediction["probability"], pt_mev, eta
    ).to_csv(output / "kinematic_slice_metrics.csv", index=False)
    return metrics


def main(argv=None):
    args = parse_args(argv)
    settings = _settings(args)
    if args.scale_neighbor > args.neighbors:
        raise ValueError("scale-neighbor cannot exceed neighbors.")
    h5_file = _resolved_path(args.h5_file)
    if not h5_file.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                "JetSet HDF5 is missing. Run download_atlas_jetset.py or add "
                "--download-if-missing: %s" % h5_file
            )
        expected_name = "mc-flavtag-ttbar-%s.h5" % args.dataset_tier
        if h5_file.name != expected_name:
            raise ValueError(
                "--h5-file basename must be %s for tier %s."
                % (expected_name, args.dataset_tier)
            )
        download_official_file(h5_file, tier=args.dataset_tier)
    inferred_tier = infer_official_tier(h5_file)
    if inferred_tier is None:
        raise ValueError("Run entry point requires an official JetSet filename.")
    if inferred_tier != args.dataset_tier:
        raise ValueError("--dataset-tier does not match --h5-file.")
    integrity = verify_official_file(h5_file, tier=inferred_tier)
    output = _resolved_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_atlas_jetset_b_vs_light(
        h5_file,
        sample=settings["sample"],
        random_state=args.random_state,
        label_definition=args.label_definition,
        protocol=args.protocol,
        max_source_events=args.max_source_events,
    )
    X, y, source_rows = dataset.X, dataset.y, dataset.row_index
    event_numbers = np.asarray(dataset.group_id, dtype=np.int64)
    source_sha256 = workspace_source_sha256()
    roles = _split_roles(y, event_numbers)
    split_audit = _role_audit(roles, y, source_rows, event_numbers)

    base = roles["base"]
    operator = roles["operator_validation"]
    dimension = roles["dimension_gate"]
    test = roles["independent_test"]
    adapter = AtlasJetSetFeatureMap(dataset.feature_names)
    X_base = adapter.fit_transform(X[base])
    X_operator = adapter.transform(X[operator])
    X_dimension = adapter.transform(X[dimension])

    print("Dataset:", dataset.dataset_name)
    print("Rows/adapter features:", X.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
    print("Unique events:", len(np.unique(event_numbers)))
    print("Analysis protocol:", args.protocol)
    event_audit = dataset.schema_audit["protocol_audit"]
    print(
        "Official/source events: %d; requested cap=%d; cap binding=%s"
        % (
            dataset.schema_audit["official_eventwise_rows"],
            event_audit["requested_max_source_events"],
            event_audit["source_event_cap_binding"],
        )
    )
    print(
        "Locked roles: base=%d operator-validation=%d dimension-gate=%d test=%d"
        % (len(base), len(operator), len(dimension), len(test))
    )
    print("Method: h_theta + graph + J/V_PI + H + orbitals + dJ/dgate + PSD")
    print("Truth/GN2/DL1 fields are audit-only and are not passed to POR.")
    print("No tree, fusion, calibration, or neural final probability.")

    model = DerivativeGatedNeuralPOR(
        n_neighbors=args.neighbors,
        scale_neighbor=args.scale_neighbor,
        lambda_pi_grid=settings["lambda_pi_grid"],
        k_max=settings["k_max"],
        dimension_bootstrap_repetitions=settings[
            "dimension_bootstrap_repetitions"
        ],
        neural_max_epochs=settings["neural_max_epochs"],
        neural_patience=settings["neural_patience"],
        measurement_rank=4,
        measurement_max_iter=settings["measurement_max_iter"],
        measurement_restarts=settings["measurement_restarts"],
        max_measurement_samples=settings["measurement_sample"],
        max_graph_samples=settings["graph_sample"],
        query_batch_size=args.query_batch_size,
        random_state=args.random_state,
    ).fit(
        X_base,
        y[base],
        X_operator_validation=X_operator,
        y_operator_validation=y[operator],
        X_dimension_validation=X_dimension,
        y_dimension_validation=y[dimension],
        X_reference_train=X_base,
        X_reference_dimension=X_dimension,
    )

    _save_model_paths(model, output)
    _save_core_artifacts(model, source_rows[base], output, args)
    adapter_summary = adapter.summary()
    measurement = measurement_summary(model)
    save_json(output / "input_feature_adapter.json", adapter_summary)
    save_json(output / "jetset_schema_and_leakage_audit.json", dataset.schema_audit)
    save_json(output / "official_file_integrity.json", integrity)
    save_json(output / "role_split_audit.json", split_audit)
    save_json(output / "neural_geometry.json", model.neural_geometry_.summary())
    save_json(output / "selected_derivative_gated_dimension.json", model.selected_)
    save_json(output / "measurement_summary.json", measurement)
    save_json(
        output / "compactness_report.json",
        compactness_report(
            "ATLAS-JetSet-b-light",
            len(dataset.feature_names),
            len(dataset.feature_names),
            model.selected_,
        ),
    )
    for class_index, operator_matrix in enumerate(model.measurement_.M_):
        pd.DataFrame(operator_matrix).to_csv(
            output / ("measurement_operator_class_%d.csv" % class_index), index=False
        )

    measurement_success = bool(model.measurement_.optimization_result_.success)
    dimension_resolved = bool(model.criterion_reached_)
    method_resolved = bool(measurement_success and dimension_resolved)
    method_status = {
        "method_resolved_before_test": method_resolved,
        "method_variant": "Derivative-Gated-Neural-POR-J-extension Unified V9",
        "physics_task": "bottom-versus-light jet flavour tagging",
        "truth_label_definition": {"0": "light jet", "1": "bottom jet"},
        "truth_label_source_audit_only": dataset.label_field,
        "dataset": "ATLAS JetSet ttbar 13.6 TeV",
        "analysis_protocol": args.protocol,
        "paper_ttbar_fiducial_selection_applied": bool(
            args.protocol == "paper-ttbar"
        ),
        "max_source_events": int(args.max_source_events),
        "event_split_protocol": split_audit["protocol"],
        "event_leakage_detected": False,
        "truth_or_precomputed_tagger_fields_used_as_input": False,
        "pdf_operator_preserved": True,
        "operator": "H=L_sym+lambda_pi*V_PI",
        "neural_network_role": "geometry map h_theta only",
        "neural_auxiliary_probability_used_for_final_prediction": False,
        "derivative_target": "J=KL(P(Y|x)||P(Y))",
        "classification_loss_derivative_used": False,
        "K_DG": int(model.selected_k_),
        "candidate_K_PF": int(model.candidate_bank_k_),
        "candidate_bank_resolved": bool(model.candidate_bank_resolved_),
        "selected_orbitals_one_based": [
            int(value + 1) for value in model.selected_orbital_indices_
        ],
        "automatic_dimension_resolved": dimension_resolved,
        "measurement_optimizer_success": measurement_success,
        "test_labels_used_for_selection": False,
        "test_scoring_deferred_by_user": bool(args.defer_test_scoring),
        "tree_model": False,
        "fusion": False,
        "probability_calibration": "none",
    }
    save_json(output / "method_status.json", method_status)

    np.savez_compressed(
        output / "frozen_split_indices.npz",
        sampled_source_rows=source_rows,
        sampled_event_numbers=event_numbers,
        base=base,
        operator_validation=operator,
        dimension_gate=dimension,
        independent_test=test,
    )
    bundle = {
        "input_feature_adapter": adapter,
        "neural_por_model": model,
        "dataset": "atlas-jetset-b-light",
        "dataset_tier": inferred_tier,
        "official_file_name": h5_file.name,
        "official_file_size": int(integrity["size_bytes"]),
        "official_file_adler32": integrity["adler32"],
        "label_definition": args.label_definition,
        "label_field": dataset.label_field,
        "analysis_protocol": args.protocol,
        "max_source_events": int(args.max_source_events),
        "sample": int(settings["sample"]),
        "random_state": int(args.random_state),
        "sampled_source_rows": source_rows,
        "sampled_event_numbers": event_numbers,
        "independent_test_indices": test,
        "source_tree_sha256": source_sha256,
        "feature_names": dataset.feature_names,
        "method_resolved_before_test": method_resolved,
    }
    joblib.dump(bundle, output / "trained_atlas_jetset_dg_npor_v9.joblib", compress=3)

    metrics = None
    if method_resolved and not args.defer_test_scoring:
        metrics = _score_test(
            model,
            adapter,
            X[test],
            y[test],
            source_rows[test],
            event_numbers[test],
            dataset.audit_variables["jet_pt_btagJes_MeV"][test],
            dataset.audit_variables["jet_eta_btagJes"][test],
            output,
            settings,
            args,
        )
        save_json(
            output / "test_scoring_status.json",
            {"scored": True, "reason": "method resolved before locked-test evaluation"},
        )
        print("\nIndependent locked-test result:")
        print(pd.DataFrame([metrics]).to_string(index=False))
    else:
        reason = (
            "explicitly deferred; run score_locked_test.py after reviewing frozen outputs"
            if args.defer_test_scoring and method_resolved
            else "automatic dimension or PSD measurement optimization unresolved"
        )
        save_json(
            output / "test_scoring_status.json",
            {
                "scored": False,
                "reason": reason,
                "dimension_resolved": dimension_resolved,
                "measurement_optimizer_success": measurement_success,
            },
        )
        print("No locked-test score:", reason)

    run_config = vars(args).copy()
    run_config.update(settings)
    save_json(output / "run_config.json", run_config)
    save_json(
        output / "reproducibility_manifest.json",
        {
            "dataset": dataset.dataset_name,
            "official_file": integrity,
            "CERN_open_data_record": "ATLAS JetSet record 93940",
            "rows_used": int(len(X)),
            "unique_events_used": int(len(np.unique(event_numbers))),
            "analysis_protocol": args.protocol,
            "max_source_events": int(args.max_source_events),
            "source_event_cap_binding": bool(
                dataset.schema_audit["protocol_audit"]["source_event_cap_binding"]
            ),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(y, return_counts=True))
            },
            "raw_features": dataset.feature_names,
            "input_feature_adapter": adapter_summary,
            "schema_and_leakage_audit": dataset.schema_audit,
            "split_audit": split_audit,
            "source_tree_sha256": source_sha256,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "dataset_reference": "10.7483/OPENDATA.ATLAS.QG8W.TO8P",
            "GN2_paper": "Nature Communications 17, 541 (2026), doi:10.1038/s41467-025-65059-6",
        },
    )

    print("\nDG-NPOR V9 completed")
    print("Neural geometry:", model.neural_geometry_.summary())
    print("Candidate K_PF:", model.candidate_bank_k_)
    print("Selected K_DG:", model.selected_k_)
    print(
        "Selected orbitals (one-based):",
        [int(value + 1) for value in model.selected_orbital_indices_],
    )
    print("Outputs:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
