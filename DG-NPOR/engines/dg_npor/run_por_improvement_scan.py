#!/usr/bin/env python3
"""Tune a frozen ATLAS JetSet DG-NPOR basis at the 70% b-efficiency OP.

The independent eventNumber-modulo-10 test role is only copied into the frozen
bundle.  No test feature, label, audit score, or prediction is accessed here.
Use score_locked_test.py exactly once after reviewing the development scan.
"""

from __future__ import annotations

import argparse
import copy
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

from examples._v9_experiment_backend import index_sha256, save_json
from por_core.working_point_scan import (
    degeneracy_safe_ranked_subset,
    derivative_result_for_bank,
    eventwise_development_split,
    extend_frozen_orbitals,
    fit_structured_readout,
    parse_positive_int_grid,
    selected_state,
    stratified_take,
)
from por_hep.atlas_jetset import load_atlas_jetset_b_vs_light, verify_official_file
from por_hep.flavour_metrics import background_rejection_at_signal_efficiency
from run_atlas_jetset_por import workspace_source_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Development-only K_PF/K_DG and hard-negative scan using a frozen "
            "DG-NPOR geometry/operator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "baseline_output",
        help="resolved prior run containing trained_atlas_jetset_dg_npor_v9.joblib",
    )
    parser.add_argument("--h5-file", required=True, help="same official JetSet HDF5")
    parser.add_argument("--output-dir", required=True, help="new or empty output directory")
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--k-pf-grid", default=None, help="comma-separated fixed candidate banks")
    parser.add_argument("--k-dg-grid", default=None, help="comma-separated requested sparse sizes")
    parser.add_argument("--measurement-sample", type=int, default=None)
    parser.add_argument("--hard-negative-pool-size", type=int, default=None)
    parser.add_argument("--gate-sample", type=int, default=None, help="0 uses all gate rows")
    parser.add_argument(
        "--wp-validation-sample", type=int, default=None, help="0 uses all WP-validation rows"
    )
    parser.add_argument("--hard-negative-iterations", type=int, default=None)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.10)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--measurement-max-iter", type=int, default=None)
    parser.add_argument("--measurement-restarts", type=int, default=None)
    parser.add_argument(
        "--skip-unweighted-control",
        action="store_true",
        help="omit the no-hard-mining control arm",
    )
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _defaults(preset):
    if preset == "quick":
        return {
            "k_pf_grid": "8,16,24,32",
            "k_dg_grid": "3,5,8,12,16",
            "measurement_sample": 4000,
            "hard_negative_pool_size": 20000,
            "gate_sample": 20000,
            "wp_validation_sample": 40000,
            "hard_negative_iterations": 1,
            "measurement_max_iter": 800,
            "measurement_restarts": 1,
        }
    return {
        "k_pf_grid": "8,16,24,32,48",
        "k_dg_grid": "3,5,8,12,16,24",
        "measurement_sample": 8000,
        "hard_negative_pool_size": 50000,
        "gate_sample": 50000,
        "wp_validation_sample": 100000,
        "hard_negative_iterations": 2,
        "measurement_max_iter": 1200,
        "measurement_restarts": 2,
    }


def _resolved_settings(args):
    defaults = _defaults(args.preset)
    values = {}
    for name in (
        "measurement_sample",
        "hard_negative_pool_size",
        "gate_sample",
        "wp_validation_sample",
        "hard_negative_iterations",
        "measurement_max_iter",
        "measurement_restarts",
    ):
        values[name] = (
            defaults[name] if getattr(args, name) is None else int(getattr(args, name))
        )
    values["k_pf_grid"] = parse_positive_int_grid(
        defaults["k_pf_grid"] if args.k_pf_grid is None else args.k_pf_grid
    )
    values["k_dg_grid"] = parse_positive_int_grid(
        defaults["k_dg_grid"] if args.k_dg_grid is None else args.k_dg_grid
    )
    return values


def _require_both_classes(name, indices, labels):
    values = np.unique(np.asarray(labels)[np.asarray(indices, dtype=int)])
    if set(values.tolist()) != {0, 1}:
        raise RuntimeError("%s does not contain both classes." % name)


def _geometry(model, adapter, dataset, indices):
    adapted = adapter.transform(dataset.X[np.asarray(indices, dtype=int)])
    return model.neural_geometry_.transform(adapted)


def _best_candidate_key(candidate):
    row = candidate["row"]
    return (
        -float(row["light_rejection_lower_68"]),
        -float(row["light_rejection"]),
        int(row["actual_K_DG"]),
        int(row["candidate_K_PF"]),
        bool(row["hard_negative_enabled"]),
    )


def main(argv=None):
    args = parse_args(argv)
    settings = _resolved_settings(args)
    baseline_output = _resolve(args.baseline_output)
    h5_file = _resolve(args.h5_file)
    output = _resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)

    bundle_path = baseline_output / "trained_atlas_jetset_dg_npor_v9.joblib"
    split_path = baseline_output / "frozen_split_indices.npz"
    if not bundle_path.exists() or not split_path.exists():
        raise FileNotFoundError("Baseline output is missing its frozen model or split indices.")
    baseline_bundle = joblib.load(bundle_path)
    baseline_model = baseline_bundle["neural_por_model"]
    adapter = baseline_bundle["input_feature_adapter"]
    required_attributes = (
        "neural_geometry_",
        "core_",
        "selection_training_indices_",
        "screening_rho",
    )
    missing = [name for name in required_attributes if not hasattr(baseline_model, name)]
    if missing:
        raise RuntimeError("Baseline model lacks frozen scan state: %s" % missing)

    integrity = verify_official_file(h5_file, tier=baseline_bundle["dataset_tier"])
    if h5_file.name != baseline_bundle["official_file_name"]:
        raise RuntimeError("JetSet filename differs from the baseline run.")
    if int(integrity["size_bytes"]) != int(baseline_bundle["official_file_size"]):
        raise RuntimeError("JetSet size differs from the baseline run.")
    if integrity["adler32"] != baseline_bundle["official_file_adler32"]:
        raise RuntimeError("JetSet Adler-32 differs from the baseline run.")

    dataset = load_atlas_jetset_b_vs_light(
        h5_file,
        sample=int(baseline_bundle["sample"]),
        random_state=int(baseline_bundle["random_state"]),
        label_definition=baseline_bundle["label_definition"],
        protocol=baseline_bundle.get("analysis_protocol", "pilot-b-light"),
        max_source_events=int(baseline_bundle.get("max_source_events", 2000000)),
    )
    if not np.array_equal(dataset.row_index, baseline_bundle["sampled_source_rows"]):
        raise RuntimeError("Sampled source rows differ from the baseline run.")
    if not np.array_equal(dataset.group_id, baseline_bundle["sampled_event_numbers"]):
        raise RuntimeError("Sampled event numbers differ from the baseline run.")

    frozen = np.load(split_path)
    base = np.asarray(frozen["base"], dtype=int)
    operator = np.asarray(frozen["operator_validation"], dtype=int)
    dimension = np.asarray(frozen["dimension_gate"], dtype=int)
    independent_test = np.asarray(frozen["independent_test"], dtype=int)
    if not np.array_equal(independent_test, baseline_bundle["independent_test_indices"]):
        raise RuntimeError("Baseline bundle and split file disagree on locked test indices.")

    random_state = (
        int(baseline_bundle["random_state"])
        if args.random_state is None
        else int(args.random_state)
    )
    derivative_gate, wp_validation = eventwise_development_split(
        dimension, dataset.group_id, random_state=random_state + 801
    )
    if settings["gate_sample"] > 0:
        derivative_gate = stratified_take(
            derivative_gate,
            dataset.y,
            settings["gate_sample"],
            random_state + 811,
        )
    if settings["wp_validation_sample"] > 0:
        wp_validation = stratified_take(
            wp_validation,
            dataset.y,
            settings["wp_validation_sample"],
            random_state + 821,
        )

    graph_positions = np.asarray(baseline_model.selection_training_indices_, dtype=int)
    if np.max(graph_positions) >= len(base):
        raise RuntimeError("Frozen graph positions are incompatible with the base role.")
    base_mask = np.ones(len(base), dtype=bool)
    base_mask[graph_positions] = False
    unused_base = base[base_mask]
    measurement_train = stratified_take(
        unused_base,
        dataset.y,
        settings["measurement_sample"],
        random_state + 831,
    )
    remaining_base = np.setdiff1d(unused_base, measurement_train, assume_unique=False)
    mining_pool = stratified_take(
        remaining_base,
        dataset.y,
        settings["hard_negative_pool_size"],
        random_state + 841,
    )

    for name, indices in (
        ("measurement_train", measurement_train),
        ("hard_negative_mining", mining_pool),
        ("derivative_gate", derivative_gate),
        ("working_point_validation", wp_validation),
    ):
        _require_both_classes(name, indices, dataset.y)
    protected_overlap = np.intersect1d(
        np.concatenate([measurement_train, mining_pool, derivative_gate, wp_validation]),
        independent_test,
    )
    if len(protected_overlap):
        raise RuntimeError("Locked-test leakage into the improvement scan.")

    max_requested_bank = min(
        max(settings["k_pf_grid"]),
        baseline_model.core_.eigensystem.phi.shape[1],
    )
    if max_requested_bank < 2:
        raise RuntimeError("Frozen operator has no usable orbital bank.")

    print("Dataset:", dataset.dataset_name)
    print("Frozen baseline:", baseline_output)
    print("Rows/features:", dataset.X.shape)
    print(
        "Development roles: readout=%d mining=%d derivative-gate=%d WP-validation=%d locked-test=%d"
        % (
            len(measurement_train),
            len(mining_pool),
            len(derivative_gate),
            len(wp_validation),
            len(independent_test),
        )
    )
    print("The locked test is not opened by this script.")

    raw_sets = {}
    stable_prefixes = {}
    for name, indices in (
        ("measurement_train", measurement_train),
        ("hard_negative_mining", mining_pool),
        ("derivative_gate", derivative_gate),
        ("working_point_validation", wp_validation),
    ):
        geometry = _geometry(baseline_model, adapter, dataset, indices)
        raw, stable_prefix = extend_frozen_orbitals(
            baseline_model, geometry, max_requested_bank
        )
        raw_sets[name] = raw
        stable_prefixes[name] = stable_prefix
    common_stable_prefix = min(stable_prefixes.values())
    bank_grid = tuple(
        value for value in settings["k_pf_grid"] if value <= common_stable_prefix
    )
    if not bank_grid:
        raise RuntimeError(
            "No requested K_PF lies inside the common stable extension prefix %d."
            % common_stable_prefix
        )

    scan_rows = []
    ranking_rows = []
    hard_paths = []
    candidates = []
    y_train = dataset.y[measurement_train]
    y_mining = dataset.y[mining_pool]
    y_wp = dataset.y[wp_validation]

    baseline_indices = np.asarray(
        baseline_model.selected_orbital_indices_, dtype=int
    )
    if np.max(baseline_indices) < raw_sets["working_point_validation"].shape[1]:
        baseline_state = selected_state(
            raw_sets["working_point_validation"], baseline_indices
        )
        baseline_probability = baseline_model.measurement_.predict_proba(
            baseline_state
        )[:, 1]
        baseline_wp = background_rejection_at_signal_efficiency(
            y_wp, baseline_probability, 0.70, confidence_level=0.68
        )
        scan_rows.append({
            "arm_type": "frozen_baseline_reference",
            "eligible_for_selection": False,
            "candidate_K_PF": int(baseline_model.candidate_bank_k_),
            "requested_K_DG": int(len(baseline_indices)),
            "actual_K_DG": int(len(baseline_indices)),
            "selected_orbitals_one_based": ";".join(
                str(int(value + 1)) for value in baseline_indices
            ),
            "hard_negative_enabled": False,
            "hard_negative_iterations": 0,
            "hard_negative_fraction": 0.0,
            "hard_negative_weight": 1.0,
            "measurement_optimizer_success": bool(
                baseline_model.measurement_.optimization_result_.success
            ),
            "measurement_train_objective": float(
                baseline_model.measurement_.train_objective_
            ),
            **baseline_wp,
        })

    for bank_width in bank_grid:
        dimension_result = derivative_result_for_bank(
            baseline_model,
            raw_sets["derivative_gate"],
            bank_width,
        )
        for record in dimension_result.records(
            baseline_model.core_.eigensystem.eigenvalues[:bank_width]
        ):
            record = dict(record)
            record["candidate_K_PF"] = int(bank_width)
            ranking_rows.append(record)

        seen_sets = set()
        for requested_k in settings["k_dg_grid"]:
            if requested_k > bank_width:
                continue
            selected_indices = degeneracy_safe_ranked_subset(
                dimension_result, requested_k
            )
            identity = tuple(selected_indices.tolist())
            if identity in seen_sets:
                continue
            seen_sets.add(identity)
            train_state = selected_state(
                raw_sets["measurement_train"], selected_indices
            )
            mining_state = selected_state(
                raw_sets["hard_negative_mining"], selected_indices
            )
            wp_state = selected_state(
                raw_sets["working_point_validation"], selected_indices
            )
            iteration_arms = [settings["hard_negative_iterations"]]
            if not args.skip_unweighted_control and settings["hard_negative_iterations"] > 0:
                iteration_arms.insert(0, 0)
            for hard_iterations in iteration_arms:
                fit = fit_structured_readout(
                    train_state,
                    y_train,
                    mining_state,
                    y_mining,
                    rank=baseline_model.measurement_rank,
                    rho=baseline_model.measurement_rho,
                    l2=baseline_model.measurement_l2,
                    max_iter=settings["measurement_max_iter"],
                    restarts=settings["measurement_restarts"],
                    hard_negative_iterations=hard_iterations,
                    hard_negative_fraction=args.hard_negative_fraction,
                    hard_negative_weight=args.hard_negative_weight,
                    random_state=(
                        random_state
                        + 100000 * int(bank_width)
                        + 1000 * int(requested_k)
                        + int(hard_iterations)
                    ),
                )
                probability = fit.measurement.predict_proba(wp_state)[:, 1]
                working_point = background_rejection_at_signal_efficiency(
                    y_wp, probability, 0.70, confidence_level=0.68
                )
                row = {
                    "arm_type": "scan_candidate",
                    "eligible_for_selection": True,
                    "candidate_K_PF": int(bank_width),
                    "requested_K_DG": int(requested_k),
                    "actual_K_DG": int(len(selected_indices)),
                    "selected_orbitals_one_based": ";".join(
                        str(int(value + 1)) for value in selected_indices
                    ),
                    "hard_negative_enabled": bool(hard_iterations > 0),
                    "hard_negative_iterations": int(hard_iterations),
                    "hard_negative_fraction": float(args.hard_negative_fraction),
                    "hard_negative_weight": float(args.hard_negative_weight),
                    "measurement_optimizer_success": bool(
                        fit.measurement.optimization_result_.success
                    ),
                    "measurement_train_objective": float(
                        fit.measurement.train_objective_
                    ),
                    **working_point,
                }
                scan_rows.append(row)
                candidate = {
                    "row": row,
                    "measurement": fit.measurement,
                    "selected_indices": selected_indices,
                    "dimension_result": dimension_result,
                    "training_path": fit.training_path,
                }
                candidates.append(candidate)
                for path_row in fit.training_path:
                    tagged = dict(path_row)
                    tagged.update({
                        "candidate_K_PF": int(bank_width),
                        "requested_K_DG": int(requested_k),
                        "actual_K_DG": int(len(selected_indices)),
                        "hard_negative_arm": bool(hard_iterations > 0),
                    })
                    hard_paths.append(tagged)
                print(
                    "KPF=%d KDG=%d hard=%s: Rlight@70=%.3f lower68=%.3f mistags=%d/%d success=%s"
                    % (
                        bank_width,
                        len(selected_indices),
                        bool(hard_iterations > 0),
                        working_point["light_rejection"],
                        working_point["light_rejection_lower_68"],
                        working_point["light_mistagged_jets"],
                        working_point["total_light_jets"],
                        fit.measurement.optimization_result_.success,
                    )
                )

    successful = [
        candidate
        for candidate in candidates
        if candidate["row"]["measurement_optimizer_success"]
    ]
    if not successful:
        raise RuntimeError("No scan arm produced a resolved PSD measurement optimizer.")
    selected = min(successful, key=_best_candidate_key)
    selected_row = dict(selected["row"])
    for row in scan_rows:
        row["selected_by_development"] = bool(row is selected["row"])

    selected_model = copy.deepcopy(baseline_model)
    selected_model.selected_orbital_indices_ = np.asarray(
        selected["selected_indices"], dtype=int
    )
    selected_model.selected_k_ = int(len(selected_model.selected_orbital_indices_))
    selected_model.k_dg_ = int(selected_model.selected_k_)
    selected_model.required_orbital_bank_ = int(
        np.max(selected_model.selected_orbital_indices_) + 1
    )
    selected_model.candidate_bank_k_ = int(selected_row["candidate_K_PF"])
    selected_model.candidate_bank_resolved_ = True
    selected_model.criterion_reached_ = True
    selected_model.measurement_ = selected["measurement"]
    selected_model.A_train_ = selected_state(
        raw_sets["measurement_train"], selected_model.selected_orbital_indices_
    )
    selected_model.y_train_encoded_ = np.asarray(y_train, dtype=int)
    selected_model.n_measurement_samples_ = int(len(y_train))
    selected_model.measurement_training_indices_ = np.asarray(
        measurement_train, dtype=int
    )
    selected_model.measurement_disjoint_from_landmarks_ = True
    selected_model.working_point_optimization_ = {
        "selection_metric": "maximum lower 68% confidence bound of R_light at 70% b-efficiency",
        "selection_split": "event-disjoint subset of official modulo-10 residue 8",
        "test_labels_or_scores_used": False,
        **selected_row,
    }
    selected_model.hard_negative_training_path_ = tuple(selected["training_path"])
    selected_model.selected_ = {
        "method_variant": "Working-Point-Optimized Derivative-Gated Neural POR",
        "pdf_core_preserved": True,
        "frozen_components": "h_theta + graph + J/V_PI + H + orbital basis",
        "candidate_bank_method": "fixed K_PF development grid",
        "dimension_extension": "dJ/dgate ranked degeneracy-safe sparse orbital subsets",
        "dimension_selected_by_validation_performance": True,
        "selection_metric": selected_model.working_point_optimization_["selection_metric"],
        "candidate_K_PF": int(selected_model.candidate_bank_k_),
        "K_DG": int(selected_model.selected_k_),
        "selected_K": int(selected_model.selected_k_),
        "selected_orbitals_one_based": [
            int(value + 1) for value in selected_model.selected_orbital_indices_
        ],
        "required_orbital_bank": int(selected_model.required_orbital_bank_),
        "reference_dimension": int(selected_model.reference_dimension_),
        "strict_reference_compression": bool(
            selected_model.selected_k_ < selected_model.reference_dimension_
        ),
        "compression_ratio_to_reference": float(
            selected_model.selected_k_ / selected_model.reference_dimension_
        ),
        "final_state": "normalized_selected_orbitals_only",
        "final_readout": "structured_PSD_with_cross_orbital_terms",
        "probability_calibration": "none",
        "neural_auxiliary_probability_used_for_final_prediction": False,
        "GN2_or_DL1_scores_used": False,
        "hard_negative_training": bool(selected_row["hard_negative_enabled"]),
        "test_labels_used_for_selection": False,
    }

    pd.DataFrame(scan_rows).to_csv(output / "working_point_scan.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(
        output / "derivative_orbital_rankings_by_bank.csv", index=False
    )
    pd.DataFrame(hard_paths).to_csv(
        output / "hard_negative_training_paths.csv", index=False
    )
    save_json(output / "selected_working_point_model.json", selected_model.working_point_optimization_)
    save_json(output / "selected_derivative_gated_dimension.json", selected_model.selected_)
    save_json(output / "official_file_integrity.json", integrity)
    save_json(output / "input_feature_adapter.json", adapter.summary())
    save_json(
        output / "development_split_audit.json",
        {
            "event_split_protocol": "official mod10; residue 9 remains locked test",
            "scan_roles": {
                "measurement_train": {
                    "rows": int(len(measurement_train)),
                    "source_index_sha256": index_sha256(dataset.row_index[measurement_train]),
                },
                "hard_negative_mining": {
                    "rows": int(len(mining_pool)),
                    "source_index_sha256": index_sha256(dataset.row_index[mining_pool]),
                },
                "derivative_gate": {
                    "rows": int(len(derivative_gate)),
                    "unique_events": int(len(np.unique(dataset.group_id[derivative_gate]))),
                    "source_index_sha256": index_sha256(dataset.row_index[derivative_gate]),
                },
                "working_point_validation": {
                    "rows": int(len(wp_validation)),
                    "unique_events": int(len(np.unique(dataset.group_id[wp_validation]))),
                    "source_index_sha256": index_sha256(dataset.row_index[wp_validation]),
                },
                "independent_test": {
                    "rows": int(len(independent_test)),
                    "source_index_sha256": index_sha256(dataset.row_index[independent_test]),
                    "opened_by_scan": False,
                },
            },
            "derivative_gate_labels_used_by_dJ_dgate": False,
            "working_point_validation_labels_used_for_model_selection": True,
            "working_point_validation_used_for_readout_training": False,
            "test_features_used": False,
            "test_labels_used": False,
            "GN2_DL1_audit_scores_used": False,
            "cross_role_event_overlap_gate_vs_wp": int(
                len(np.intersect1d(
                    dataset.group_id[derivative_gate], dataset.group_id[wp_validation]
                ))
            ),
            "common_inductive_stable_prefix": int(common_stable_prefix),
            "stable_prefixes": stable_prefixes,
        },
    )

    np.savez_compressed(
        output / "frozen_split_indices.npz",
        sampled_source_rows=dataset.row_index,
        sampled_event_numbers=dataset.group_id,
        base=base,
        operator_validation=operator,
        dimension_gate=dimension,
        derivative_gate=derivative_gate,
        working_point_validation=wp_validation,
        measurement_train=measurement_train,
        hard_negative_mining=mining_pool,
        independent_test=independent_test,
    )

    method_resolved = bool(selected_model.measurement_.optimization_result_.success)
    source_sha256 = workspace_source_sha256()
    final_bundle = dict(baseline_bundle)
    final_bundle.update({
        "neural_por_model": selected_model,
        "method_resolved_before_test": method_resolved,
        "source_tree_sha256": source_sha256,
        "independent_test_indices": independent_test,
        "improvement_protocol": "KPF-KDG-Rlight70-hard-negative-development-scan",
        "baseline_source_tree_sha256": baseline_bundle.get("source_tree_sha256"),
        "working_point_selection": selected_model.working_point_optimization_,
    })
    joblib.dump(
        final_bundle,
        output / "trained_atlas_jetset_dg_npor_v9.joblib",
        compress=3,
    )
    for class_index, operator_matrix in enumerate(selected_model.measurement_.M_):
        pd.DataFrame(operator_matrix).to_csv(
            output / ("measurement_operator_class_%d.csv" % class_index), index=False
        )
    save_json(
        output / "method_status.json",
        {
            "method_resolved_before_test": method_resolved,
            "method_variant": selected_model.selected_["method_variant"],
            "analysis_protocol": baseline_bundle.get("analysis_protocol"),
            "candidate_K_PF": int(selected_model.candidate_bank_k_),
            "K_DG": int(selected_model.selected_k_),
            "selected_orbitals_one_based": selected_model.selected_[
                "selected_orbitals_one_based"
            ],
            "selection_metric": selected_model.selected_["selection_metric"],
            "hard_negative_training": bool(selected_row["hard_negative_enabled"]),
            "truth_or_precomputed_tagger_fields_used_as_input": False,
            "test_labels_used_for_selection": False,
            "test_scoring_deferred_by_protocol": True,
            "tree_model": False,
            "fusion": False,
            "probability_calibration": "none",
        },
    )
    run_config = vars(args).copy()
    run_config.update({
        key: list(value) if isinstance(value, tuple) else value
        for key, value in settings.items()
    })
    run_config["bootstrap_repetitions"] = int(args.bootstrap_repetitions)
    save_json(output / "run_config.json", run_config)
    save_json(
        output / "test_scoring_status.json",
        {
            "scored": False,
            "reason": "locked test deferred until development-only scan was frozen",
            "test_features_opened_by_scan": False,
            "test_labels_opened_by_scan": False,
        },
    )
    save_json(
        output / "reproducibility_manifest.json",
        {
            "source_tree_sha256": source_sha256,
            "baseline_source_tree_sha256": baseline_bundle.get("source_tree_sha256"),
            "official_file": integrity,
            "sample": int(baseline_bundle["sample"]),
            "random_state": int(random_state),
            "analysis_protocol": baseline_bundle.get("analysis_protocol"),
            "selection_metric": selected_model.selected_["selection_metric"],
            "test_labels_used": False,
            "GN2_DL1_scores_used": False,
        },
    )

    print("\nSelected development model:")
    print(pd.DataFrame([selected_row]).to_string(index=False))
    print("\nLocked test remains closed.")
    print("Next: score_locked_test.py", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
