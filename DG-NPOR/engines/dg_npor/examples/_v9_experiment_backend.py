#!/usr/bin/env python3
"""Run derivative-gated Neural-POR without external baselines."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from sklearn.model_selection import train_test_split

from por_core import DerivativeGatedNeuralPOR
from por_hep import (
    CompactATLASPhysicsFeatureMap,
    DenseNet201DirectFeatureMap,
    SUSYPhysicsFeatureMap,
    fixed_size_role_indices,
    grouped_locked_split_indices,
    load_particle_dataset,
    locked_split_indices,
    stratified_group_holdout_indices,
)
from por_hep.metrics import evaluate_raw_probability, stratified_bootstrap_intervals


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def index_sha256(values):
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(project_root):
    digest = hashlib.sha256()
    paths = []
    for relative_root in ("src", "examples"):
        paths.extend((Path(project_root) / relative_root).rglob("*.py"))
    for path in sorted(paths, key=lambda item: str(item)):
        relative = str(path.relative_to(project_root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def make_input_adapter(dataset, feature_names, variant):
    if variant == "atlas-compact":
        if dataset != "atlas-higgs":
            raise ValueError("atlas-compact requires dataset atlas-higgs.")
        return CompactATLASPhysicsFeatureMap(feature_names)
    if variant == "susy-physics":
        if dataset != "susy":
            raise ValueError("susy-physics requires dataset susy.")
        return SUSYPhysicsFeatureMap(feature_names)
    if variant == "densenet201-direct":
        if dataset != "densenet201":
            raise ValueError("densenet201-direct requires dataset densenet201.")
        return DenseNet201DirectFeatureMap(feature_names)
    raise ValueError("Unknown or incompatible input-adapter variant.")


def experiment_defaults(dataset):
    table = {
        "susy": {
            "sample": 20000,
            "neighbors": 32,
            "scale_neighbor": 16,
            "output_dir": "results_susy_derivative_gated_v9",
        },
        "atlas-higgs": {
            "sample": 20000,
            "neighbors": 64,
            "scale_neighbor": 32,
            "output_dir": "results_atlas_higgs_compact_v9",
        },
        "densenet201": {
            "sample": 0,
            "neighbors": 15,
            "scale_neighbor": 8,
            "output_dir": "results_densenet201_derivative_gated_v9",
        },
    }
    if dataset not in table:
        raise ValueError("No experiment defaults exist for dataset %r." % dataset)
    return table[dataset]


def adapter_summary(adapter, variant):
    if hasattr(adapter, "summary"):
        return adapter.summary()
    return {
        "name": "%s:%s" % (type(adapter).__name__, variant),
        "labels_used": False,
        "output_dimension": int(len(adapter.output_feature_names_)),
    }


def grouping_audit(dataset, role_indices):
    groups = np.asarray(dataset.group_id)
    names = list(role_indices)
    intersections = {}
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            overlap = np.intersect1d(
                groups[role_indices[first_name]],
                groups[role_indices[second_name]],
            )
            intersections["%s__%s" % (first_name, second_name)] = int(
                len(overlap)
            )
            if len(overlap):
                raise RuntimeError("Group leakage detected between split roles.")
    _, counts = np.unique(groups, return_counts=True)
    return {
        "grouping_source": dataset.grouping_source,
        "rows": int(len(groups)),
        "unique_groups": int(len(counts)),
        "duplicated_groups": int(np.sum(counts > 1)),
        "largest_group_rows": int(np.max(counts)),
        "cross_role_group_intersections": intersections,
        "group_leakage_detected": False,
    }


def operator_grid(preset):
    if preset == "quick":
        return (0.10, 0.25, 0.50, 1.0)
    return (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


def measurement_summary(model):
    operators = np.asarray(model.measurement_.M_, dtype=float)
    records = []
    for class_index, operator in enumerate(operators):
        diagonal = np.diag(np.diag(operator))
        off_diagonal = operator - diagonal
        records.append({
            "class_index": int(class_index),
            "matrix_dimension": int(operator.shape[0]),
            "rank_parameter": int(model.measurement_.rank_),
            "off_diagonal_frobenius_norm": float(np.linalg.norm(off_diagonal)),
            "off_diagonal_fraction": float(
                np.linalg.norm(off_diagonal)
                / max(np.linalg.norm(operator), 1e-12)
            ),
            "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(operator))),
        })
    completeness = operators.sum(axis=0)
    return {
        "parallel_connection": "off-diagonal M_ij cross-orbital terms",
        "operators": records,
        "sum_operator_identity_max_abs_error": float(
            np.max(np.abs(completeness - np.eye(completeness.shape[0])))
        ),
        "optimizer_success": bool(model.measurement_.optimization_result_.success),
        "optimizer_message": str(model.measurement_.optimization_result_.message),
        "optimizer_iterations": int(model.measurement_.n_iterations_),
        "optimizer_gradient_norm": float(model.measurement_.gradient_norm_),
        "measurement_train_objective": float(model.measurement_.train_objective_),
    }


def compactness_report(dataset, raw_dimension, adapted_dimension, selected):
    K = int(selected["K_DG"])
    geometry_dimension = int(
        selected["neural_geometry"]["geometry_dimension"]
    )
    dimensions = {
        "raw_input": int(raw_dimension),
        "adapted_feature_input": int(adapted_dimension),
        "neural_geometry_h_theta": geometry_dimension,
        "normalized_orbital_state": K,
    }
    return {
        "dataset": dataset,
        "dimension_method": "derivative_gated_PDF_predictive_information",
        "K_DG": K,
        "selected_orbitals_one_based": selected["selected_orbitals_one_based"],
        "required_orbital_bank": int(selected["required_orbital_bank"]),
        "K_or_delta_supplied_by_user": False,
        "dimensions": dimensions,
        "orbital_to_input_ratios": {
            name: float(K / dimension)
            for name, dimension in dimensions.items()
            if name != "normalized_orbital_state"
        },
        "strict_dimension_reduction": {
            name: bool(K < dimension)
            for name, dimension in dimensions.items()
            if name != "normalized_orbital_state"
        },
        "interpretation": (
            "h_theta estimates task-aware event geometry and is not the compact "
            "claim. K_DG is the number of non-contiguous orbitals selected "
            "from dJ/dgate; the largest orbital index is not the dimension."
        ),
    }


def predict_batches(model, physics, X, batch_size):
    output = {name: [] for name in (
        "probability", "diagonal_only_probability",
        "diagonal_signal_contribution", "cross_orbital_signal_contribution",
    )}
    size = max(1, int(batch_size))
    for start in range(0, len(X), size):
        physics_input = physics.transform(X[start:start + size])
        decomposition = model.probability_decomposition(physics_input)
        output["probability"].append(decomposition["probability"][:, 1])
        output["diagonal_only_probability"].append(
            decomposition["diagonal_only_probability"][:, 1]
        )
        output["diagonal_signal_contribution"].append(
            decomposition["diagonal"][:, 1]
        )
        output["cross_orbital_signal_contribution"].append(
            decomposition["cross_orbital"][:, 1]
        )
    return {name: np.concatenate(values) for name, values in output.items()}


def main(argv=None, fixed_dataset=None, fixed_feature_map=None):
    if fixed_dataset is None or fixed_feature_map is None:
        raise ValueError(
            "Use one of the three dataset-specific example runners."
        )
    defaults = experiment_defaults(fixed_dataset)
    parser = argparse.ArgumentParser(
        description=(
            "Neural geometry + PDF POR + dJ/dgate sparse orbitals + PSD"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path")
    parser.set_defaults(dataset=fixed_dataset, feature_map=fixed_feature_map)
    if fixed_dataset == "densenet201":
        parser.add_argument(
            "--group-column",
            default=None,
            help=(
                "Optional patient/subject identifier. Without it, exact "
                "duplicate 1920-vectors are kept in one split."
            ),
        )
    else:
        parser.set_defaults(group_column=None)
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="New or empty directory for all auditable outputs.",
    )
    parser.add_argument("--sample", type=int, default=defaults["sample"],
                        help="Rows used; 0 means all rows.")
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    split_choices = (
        ("random", "uci-official") if fixed_dataset == "susy" else ("random",)
    )
    parser.add_argument(
        "--split-protocol", choices=split_choices, default="random"
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--dimension-audit-size", type=float, default=0.20)
    parser.add_argument("--operator-validation-size", type=float, default=0.20)
    parser.add_argument("--base-sample", type=int, default=0)
    parser.add_argument("--operator-validation-sample", type=int, default=0)
    parser.add_argument("--dimension-audit-sample", type=int, default=0)
    parser.add_argument(
        "--graph-sample", type=int, default=10000,
        help="Maximum base-role rows used as graph landmarks.",
    )
    parser.add_argument(
        "--neighbors", type=int, default=defaults["neighbors"],
        help="kNN graph neighborhood size.",
    )
    parser.add_argument(
        "--scale-neighbor", type=int, default=defaults["scale_neighbor"],
        help="Neighbor rank defining the self-tuning local kernel scale.",
    )
    parser.add_argument("--k-max", type=int, default=128,
                        help="Computational spectrum ceiling; never selected K.")
    parser.add_argument("--dimension-bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--screening-rho", type=float, default=1e-3)
    parser.add_argument(
        "--risk-audit-reference",
        choices=("adapted", "raw", "physics"),
        default="adapted",
        help="Audit reference only; it never selects the orbital dimension.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--probe-c", type=float, default=1.0)
    parser.add_argument("--neural-max-epochs", type=int, default=250)
    parser.add_argument("--neural-patience", type=int, default=25)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--neural-alpha", type=float, default=1e-4)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--measurement-rank", type=int, default=4)
    parser.add_argument("--measurement-sample", type=int, default=6000)
    parser.add_argument("--measurement-max-iter", type=int, default=1200)
    parser.add_argument("--measurement-restarts", type=int, default=2)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-repetitions", type=int, default=300)
    parser.add_argument("--bootstrap-sample", type=int, default=50000)
    parser.add_argument("--hash-input", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    if args.scale_neighbor > args.neighbors:
        raise ValueError("scale-neighbor cannot exceed neighbors.")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new name.")
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_particle_dataset(
        args.csv_path, args.dataset, sample=args.sample,
        random_state=args.random_state,
        group_column=args.group_column,
    )
    X_all, y_all, rows_all = dataset.X, dataset.y, dataset.row_index
    print("Dataset:", dataset.dataset_name)
    print("Rows/raw features:", X_all.shape)
    print("Class counts:", dict(zip(*np.unique(y_all, return_counts=True))))
    print("Split grouping:", dataset.grouping_source)
    print("Method: h_theta + PDF POR + dJ/dgate sparse orbitals + PSD")
    print("No baseline, tree, fusion, neural final probability or calibration.")

    if args.split_protocol == "uci-official":
        if args.dataset != "susy" or args.sample != 0 or len(X_all) != 5000000:
            raise ValueError(
                "uci-official requires the complete 5,000,000-row SUSY file."
            )
        fixed = (
            args.base_sample, args.operator_validation_sample,
            args.dimension_audit_sample,
        )
        if min(fixed) <= 0:
            raise ValueError("Official mode requires explicit positive role sizes.")

    use_grouped_split = args.dataset == "densenet201"
    if use_grouped_split:
        if args.split_protocol != "random":
            raise ValueError("DenseNet201 supports only grouped random protocol.")
        locked = grouped_locked_split_indices(
            y_all,
            dataset.group_id,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    else:
        locked = locked_split_indices(
            y_all, protocol=args.split_protocol, test_size=args.test_size,
            random_state=args.random_state,
        )
    development_indices, test_indices = locked.development, locked.test
    fixed = (
        args.base_sample, args.operator_validation_sample,
        args.dimension_audit_sample,
    )
    if any(value > 0 for value in fixed):
        if use_grouped_split:
            raise ValueError(
                "Fixed row-count roles are disabled for grouped DenseNet splits."
            )
        if not all(value > 0 for value in fixed):
            raise ValueError("Set all three fixed role samples or none.")
        roles = fixed_size_role_indices(
            development_indices, y_all, *fixed,
            random_state=args.random_state + 1,
        )
        base_indices = roles.base
        operator_indices = roles.operator_validation
        dimension_indices = roles.dimension_audit
    else:
        if use_grouped_split:
            work_indices, dimension_indices = stratified_group_holdout_indices(
                development_indices,
                y_all,
                dataset.group_id,
                test_size=args.dimension_audit_size,
                random_state=args.random_state + 1,
            )
            base_indices, operator_indices = stratified_group_holdout_indices(
                work_indices,
                y_all,
                dataset.group_id,
                test_size=args.operator_validation_size,
                random_state=args.random_state + 2,
            )
        else:
            work_indices, dimension_indices = train_test_split(
                development_indices,
                test_size=args.dimension_audit_size,
                stratify=y_all[development_indices],
                random_state=args.random_state + 1,
            )
            base_indices, operator_indices = train_test_split(
                work_indices,
                test_size=args.operator_validation_size,
                stratify=y_all[work_indices],
                random_state=args.random_state + 2,
            )
    X_base, y_base = X_all[base_indices], y_all[base_indices]
    X_operator, y_operator = X_all[operator_indices], y_all[operator_indices]
    X_dimension, y_dimension = X_all[dimension_indices], y_all[dimension_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    group_report = grouping_audit(dataset, {
        "base": base_indices,
        "operator_validation": operator_indices,
        "dimension_gate": dimension_indices,
        "independent_test": test_indices,
    })
    print(
        "Locked split sizes: base=%d operator-validation=%d dimension-gate=%d "
        "development=%d independent-test=%d"
        % (len(X_base), len(X_operator), len(X_dimension),
           len(development_indices), len(X_test))
    )

    physics = make_input_adapter(
        args.dataset, dataset.feature_names, variant=args.feature_map
    )
    X_base_p = physics.fit_transform(X_base)
    X_operator_p = physics.transform(X_operator)
    X_dimension_p = physics.transform(X_dimension)
    feature_summary = adapter_summary(physics, args.feature_map)
    print("Input adapter:", feature_summary["name"])
    print("Adapted feature dimension:", X_base_p.shape[1])
    if args.risk_audit_reference == "raw":
        reference_train, reference_dimension = X_base, X_dimension
    else:
        reference_train, reference_dimension = X_base_p, X_dimension_p

    model = DerivativeGatedNeuralPOR(
        n_neighbors=args.neighbors,
        scale_neighbor=args.scale_neighbor,
        lambda_pi_grid=operator_grid(args.preset),
        k_max=args.k_max,
        dimension_bootstrap_repetitions=args.dimension_bootstrap_repetitions,
        screening_rho=args.screening_rho,
        confidence_level=args.confidence_level,
        probe_C=args.probe_c,
        neural_max_epochs=args.neural_max_epochs,
        neural_patience=args.neural_patience,
        neural_learning_rate=args.neural_learning_rate,
        neural_alpha=args.neural_alpha,
        neural_batch_size=args.neural_batch_size,
        measurement_rank=args.measurement_rank,
        measurement_max_iter=args.measurement_max_iter,
        measurement_restarts=args.measurement_restarts,
        max_measurement_samples=args.measurement_sample,
        max_graph_samples=args.graph_sample,
        query_batch_size=args.query_batch_size,
        random_state=args.random_state,
    ).fit(
        X_base_p,
        y_base,
        X_operator_validation=X_operator_p,
        y_operator_validation=y_operator,
        X_dimension_validation=X_dimension_p,
        y_dimension_validation=y_dimension,
        X_reference_train=reference_train,
        X_reference_dimension=reference_dimension,
    )
    print("Neural geometry:", model.neural_geometry_.summary())
    print("Automatic derivative-gated dimension:", model.selected_)

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
    save_json(output / "neural_geometry.json", model.neural_geometry_.summary())
    save_json(output / "input_feature_adapter.json", feature_summary)
    save_json(output / "group_split_audit.json", group_report)
    save_json(output / "selected_derivative_gated_dimension.json", model.selected_)
    save_json(output / "compactness_report.json", compactness_report(
        args.dataset, X_base.shape[1], X_base_p.shape[1], model.selected_
    ))
    save_json(output / "measurement_summary.json", measurement_summary(model))
    joblib.dump(
        {"input_feature_adapter": physics, "neural_por_model": model,
         "dataset": args.dataset, "feature_map_variant": args.feature_map,
         "grouping_source": dataset.grouping_source},
        output / "trained_derivative_gated_neural_por.joblib",
        compress=3,
    )

    measurement_success = bool(model.measurement_.optimization_result_.success)
    dimension_resolved = bool(model.criterion_reached_)
    test_scoring_valid = bool(measurement_success and dimension_resolved)
    status = {
        "method_valid_for_test_scoring": test_scoring_valid,
        "method_variant": "Derivative-Gated-Neural-POR-J-extension",
        "feature_map_variant": args.feature_map,
        "PCA": False,
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
        "risk_audit_used_to_select_K": False,
        "test_labels_used_for_selection": False,
        "tree_model": False,
        "fusion": False,
        "probability_calibration": "none",
        "measurement_optimizer_success": measurement_success,
    }
    save_json(output / "method_status.json", status)
    for class_index, operator in enumerate(model.measurement_.M_):
        pd.DataFrame(operator).to_csv(
            output / ("measurement_operator_class_%d.csv" % class_index),
            index=False,
        )

    if test_scoring_valid:
        prediction = predict_batches(
            model, physics, X_test, args.query_batch_size
        )
        metrics = evaluate_raw_probability(
            "%s-DerivativeGated-Neural-POR" % args.dataset,
            y_test,
            prediction["probability"],
            probability_source=(
                "dJ_dgate_sparse_neural_por_psd_no_fusion_no_calibration"
            ),
        )
        diagonal = evaluate_raw_probability(
            "%s-DerivativeGated-Neural-POR-diagonal-only-ablation" % args.dataset,
            y_test,
            prediction["diagonal_only_probability"],
            probability_source=(
                "same_derivative_gated_state_diagonal_only_psd_ablation"
            ),
        )
        pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame([metrics, diagonal]).to_csv(
            output / "ablation_metrics.csv", index=False
        )
        save_json(
            output / "test_confidence_intervals.json",
            stratified_bootstrap_intervals(
                y_test,
                prediction["probability"],
                repetitions=args.bootstrap_repetitions,
                confidence_level=args.confidence_level,
                max_samples=args.bootstrap_sample,
                random_state=args.random_state + 5001,
            ),
        )
        pd.DataFrame({
            "row_index": rows_all[test_indices],
            "y_true": y_test,
            **prediction,
        }).to_csv(output / "test_predictions.csv", index=False)
        print("\nIndependent locked-test result:")
        print(pd.DataFrame([metrics]).to_string(index=False))
    else:
        print("No test score: automatic dimension or PSD optimization unresolved.")

    save_json(output / "run_config.json", vars(args))
    project_root = Path(__file__).resolve().parents[1]
    save_json(output / "reproducibility_manifest.json", {
        "dataset": dataset.dataset_name,
        "input_file": Path(args.csv_path).name,
        "input_sha256": file_sha256(args.csv_path) if args.hash_input else None,
        "source_tree_sha256": source_tree_sha256(project_root),
        "rows_used": int(len(X_all)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y_all, return_counts=True))
        },
        "raw_features": dataset.feature_names,
        "input_feature_adapter": feature_summary,
        "adapted_features": physics.output_feature_names_,
        "grouping_source": dataset.grouping_source,
        "unique_groups": int(len(np.unique(dataset.group_id))),
        "split_protocol": locked.protocol,
        "split_audit": {
            "base": {"size": int(len(base_indices)),
                     "index_sha256": index_sha256(rows_all[base_indices])},
            "operator_validation": {"size": int(len(operator_indices)),
                     "index_sha256": index_sha256(rows_all[operator_indices])},
            "dimension_audit": {"size": int(len(dimension_indices)),
                     "index_sha256": index_sha256(rows_all[dimension_indices])},
            "measurement_candidate_pool": {
                "size": int(
                    len(development_indices) - model.n_landmark_samples_
                ),
                "note": (
                    "development observations outside the frozen landmark graph"
                ),
            },
            "independent_test": {"size": int(len(test_indices)),
                     "index_sha256": index_sha256(rows_all[test_indices])},
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    })
    print("Saved results to:", output.resolve())


if __name__ == "__main__":
    main()
