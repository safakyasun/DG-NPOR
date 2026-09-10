#!/usr/bin/env python3
"""Development-only physics-constrained self-configuring NN + frozen POR.

Stage A trains a bounded family of sparse-gated structured encoders.  It uses
only event-disjoint encoder-train, early-stop, and architecture-validation
roles.  Stage B sends the best encoder candidates through the unchanged
DG-NPOR operator/orbital/PSD pipeline and selects at R_light@70 on a separate
working-point validation role.  The residue-9 locked test is never read here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import measurement_summary, save_json
from por_core import DerivativeGatedNeuralPOR
from por_core.working_point_scan import eventwise_development_split, stratified_take
from por_hep.atlas_jetset import (
    download_official_file,
    infer_official_tier,
    verify_official_file,
)
from por_hep.flavour_metrics import background_rejection_at_signal_efficiency
from por_hep.self_configuring_geometry import (
    PhysicsConstrainedGatedTrackGeometry,
    dropout_stability_audit,
)
from por_hep.structured_tracks import (
    H5StructuredTrackSource,
    IdentityFeatureMap,
    load_atlas_jetset_index,
)
from run_atlas_jetset_por import (
    _role_audit,
    _save_core_artifacts,
    _save_model_paths,
    _split_roles,
    workspace_source_sha256,
)
from run_structured_track_scan import _eventwise_encoder_split


def preset_defaults(preset):
    if preset == "quick":
        return {
            "sample": 20000,
            "encoder_sample": 24000,
            "max_tracks": 20,
            "encoder_epochs": 14,
            "encoder_patience": 3,
            "encoder_batch_size": 512,
            "architecture_top_k": 2,
            "graph_sample": 6000,
            "k_max": 64,
            "dimension_bootstrap_repetitions": 50,
            "neural_max_epochs": 100,
            "neural_patience": 12,
            "measurement_sample": 4000,
            "measurement_max_iter": 800,
            "measurement_restarts": 1,
            "lambda_pi_grid": (0.10, 0.25, 0.50, 1.0),
        }
    return {
        "sample": 500000,
        "encoder_sample": 80000,
        "max_tracks": 40,
        "encoder_epochs": 28,
        "encoder_patience": 6,
        "encoder_batch_size": 512,
        "architecture_top_k": 3,
        "graph_sample": 16000,
        "k_max": 128,
        "dimension_bootstrap_repetitions": 250,
        "neural_max_epochs": 220,
        "neural_patience": 24,
        "measurement_sample": 10000,
        "measurement_max_iter": 1500,
        "measurement_restarts": 3,
        "lambda_pi_grid": (0.05, 0.10, 0.25, 0.50, 1.0, 2.0),
    }


def architecture_library(preset):
    """Small legal search space; no arbitrary generated code or truth inputs."""
    quick = [
        dict(name="reference", hidden=24, geometry=48, depth=1, heads=1, steps=3, neighbors=6, temperature=2.0, h_theta="standard"),
        dict(name="compact_local", hidden=16, geometry=32, depth=1, heads=1, steps=2, neighbors=4, temperature=1.5, h_theta="shallow"),
        dict(name="balanced_two_head", hidden=24, geometry=48, depth=2, heads=2, steps=3, neighbors=6, temperature=2.0, h_theta="deep"),
        dict(name="global_two_head", hidden=24, geometry=48, depth=2, heads=2, steps=2, neighbors=8, temperature=3.0, h_theta="standard"),
        dict(name="wide_local", hidden=32, geometry=64, depth=2, heads=1, steps=4, neighbors=6, temperature=1.5, h_theta="tapered"),
        dict(name="deep_balanced", hidden=24, geometry=64, depth=3, heads=2, steps=3, neighbors=8, temperature=2.5, h_theta="deep"),
    ]
    if preset == "quick":
        return quick
    return quick + [
        dict(name="wide_two_head", hidden=32, geometry=64, depth=2, heads=2, steps=3, neighbors=8, temperature=2.0, h_theta="standard"),
        dict(name="compact_four_head", hidden=16, geometry=48, depth=2, heads=4, steps=3, neighbors=6, temperature=2.0, h_theta="deep"),
        dict(name="wide_four_head", hidden=32, geometry=64, depth=3, heads=4, steps=3, neighbors=8, temperature=2.5, h_theta="tapered"),
        dict(name="deep_local", hidden=32, geometry=48, depth=3, heads=2, steps=4, neighbors=4, temperature=1.5, h_theta="deep"),
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Physics-constrained self-configuring Hybrid-POR NN scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--protocol", choices=("pilot-b-light", "paper-ttbar"), default="paper-ttbar")
    parser.add_argument("--max-source-events", type=int, default=2000000)
    parser.add_argument("--dataset-tier", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--h5-file", default="data/atlas_jetset/mc-flavtag-ttbar-small.h5")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--label-definition", choices=("cone", "ghost"), default="cone")
    parser.add_argument("--encoder-sample", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--encoder-patience", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-3)
    parser.add_argument("--encoder-l2", type=float, default=1e-4)
    parser.add_argument("--gate-l0", type=float, default=2e-4)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--parallel-floor", type=float, default=0.02)
    parser.add_argument("--architecture-top-k", type=int, default=None)
    parser.add_argument("--stability-weight", type=float, default=0.15)
    parser.add_argument("--complexity-weight", type=float, default=0.005)
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
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/atlas_jetset_self_configuring_nn")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _settings(args):
    defaults = preset_defaults(args.preset)
    names = (
        "sample", "encoder_sample", "max_tracks", "encoder_epochs",
        "encoder_patience", "encoder_batch_size", "architecture_top_k",
        "graph_sample", "k_max", "dimension_bootstrap_repetitions",
        "neural_max_epochs", "neural_patience", "measurement_sample",
        "measurement_max_iter", "measurement_restarts",
    )
    values = {
        name: defaults[name] if getattr(args, name) is None else int(getattr(args, name))
        for name in names
    }
    values["lambda_pi_grid"] = defaults["lambda_pi_grid"]
    return values


def _parameter_tie_selection(arms):
    """One-percent lower-bound equivalence, then choose the simpler model."""
    best_lower = max(float(arm["working_point"]["light_rejection_lower_68"]) for arm in arms)
    equivalent = [
        arm
        for arm in arms
        if float(arm["working_point"]["light_rejection_lower_68"])
        >= 0.99 * best_lower
    ]
    return min(
        equivalent,
        key=lambda arm: (
            int(arm["encoder"].parameter_count(effective=True)),
            -float(arm["working_point"]["light_rejection"]),
            int(arm["model"].selected_k_),
            str(arm["config"]["name"]),
        ),
    )


def main(argv=None):
    args = parse_args(argv)
    settings = _settings(args)
    if args.scale_neighbor > args.neighbors:
        raise ValueError("scale-neighbor cannot exceed neighbors.")
    if settings["architecture_top_k"] < 1:
        raise ValueError("architecture-top-k must be positive.")
    h5_file = _resolve(args.h5_file)
    if not h5_file.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(h5_file)
        download_official_file(h5_file, tier=args.dataset_tier)
    tier = infer_official_tier(h5_file)
    if tier != args.dataset_tier:
        raise ValueError("--dataset-tier does not match --h5-file.")
    integrity = verify_official_file(h5_file, tier=tier)
    output = _resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_atlas_jetset_index(
        h5_file,
        sample=settings["sample"],
        random_state=args.random_state,
        label_definition=args.label_definition,
        protocol=args.protocol,
        max_source_events=args.max_source_events,
    )
    y, events, rows = dataset.y, dataset.group_id, dataset.row_index
    roles = _split_roles(y, events)
    split_audit = _role_audit(roles, y, rows, events)
    base = roles["base"]
    operator = roles["operator_validation"]
    dimension = roles["dimension_gate"]
    locked_test = roles["independent_test"]
    derivative_gate, wp_validation = eventwise_development_split(
        dimension, events, random_state=args.random_state + 501
    )

    encoder_pool = stratified_take(
        base, y, min(settings["encoder_sample"], len(base)), args.random_state + 601
    )
    encoder_fit_pool, architecture_validation = _eventwise_encoder_split(
        encoder_pool, events, y, args.random_state + 611, validation_fraction=0.20
    )
    encoder_train, encoder_early_stop = _eventwise_encoder_split(
        encoder_fit_pool, events, y, args.random_state + 621, validation_fraction=0.15
    )
    role_event_sets = [
        set(events[index].tolist())
        for index in (encoder_train, encoder_early_stop, architecture_validation)
    ]
    if any(role_event_sets[i] & role_event_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Event leakage among encoder self-configuration roles.")

    print("Dataset:", dataset.dataset_name)
    print("Selected jets:", len(y), "with up to", settings["max_tracks"], "tracks per jet")
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
    print(
        "Roles: base=%d operator=%d derivative=%d WP-validation=%d locked-test=%d"
        % (len(base), len(operator), len(derivative_gate), len(wp_validation), len(locked_test))
    )
    print(
        "Encoder roles: train=%d early-stop=%d architecture-validation=%d"
        % (len(encoder_train), len(encoder_early_stop), len(architecture_validation))
    )
    print("POR core is frozen: graph + J/V_PI + H + orbitals + dJ/dgate + PSD.")
    print("Locked-test tracks, truth, GN2, and DL1 are not opened during selection.")

    candidates = architecture_library(args.preset)
    screen_rows = []
    trained = []
    with H5StructuredTrackSource(h5_file, rows, max_tracks=settings["max_tracks"]) as source:
        for candidate_index, config in enumerate(candidates):
            print("\nArchitecture screen %d/%d: %s" % (candidate_index + 1, len(candidates), config["name"]))
            encoder = PhysicsConstrainedGatedTrackGeometry(
                hidden_dimension=config["hidden"],
                geometry_dimension=config["geometry"],
                geometry_depth=config["depth"],
                attention_heads=config["heads"],
                max_tracks=settings["max_tracks"],
                graph_neighbors=config["neighbors"],
                sequential_steps=config["steps"],
                parallel_temperature=config["temperature"],
                parallel_floor=args.parallel_floor,
                gate_l0=args.gate_l0,
                gate_threshold=args.gate_threshold,
                learning_rate=args.encoder_learning_rate,
                l2=args.encoder_l2,
                batch_size=settings["encoder_batch_size"],
                max_epochs=settings["encoder_epochs"],
                patience=settings["encoder_patience"],
                random_state=args.random_state + 1000,
            ).fit(
                source,
                encoder_train,
                y[encoder_train],
                encoder_early_stop,
                y[encoder_early_stop],
                read_batch_size=max(args.read_batch_size, 2048),
            )
            architecture_result = encoder.encode_source(
                source, architecture_validation, batch_size=args.read_batch_size
            )
            architecture_log_loss = float(
                log_loss(
                    y[architecture_validation],
                    architecture_result["auxiliary_probability"],
                    labels=[0, 1],
                )
            )
            audit_take = stratified_take(
                architecture_validation,
                y,
                min(4000, len(architecture_validation)),
                args.random_state + 2000,
            )
            stability = dropout_stability_audit(
                encoder,
                source,
                audit_take,
                random_state=args.random_state + 3000,
                read_batch_size=args.read_batch_size,
            )
            effective_parameters = encoder.parameter_count(effective=True)
            screen_score = (
                architecture_log_loss
                + args.stability_weight * stability["mean_absolute_probability_shift"]
                + args.complexity_weight * math.log1p(effective_parameters / 1000.0)
            )
            row = {
                **config,
                "architecture_validation_log_loss": architecture_log_loss,
                "dropout_mean_probability_shift": stability["mean_absolute_probability_shift"],
                "dropout_p95_probability_shift": stability["p95_absolute_probability_shift"],
                "effective_parameter_count": effective_parameters,
                "screen_score_lower_is_better": float(screen_score),
                "selected_branches": ";".join(encoder.summary()["selected_branches"]),
                "locked_test_used": False,
            }
            screen_rows.append(row)
            trained.append({"config": config, "encoder": encoder, "screen": row})
            pd.DataFrame(encoder.history_).to_csv(
                output / ("encoder_history_%s.csv" % config["name"]), index=False
            )
            save_json(output / ("encoder_%s.json" % config["name"]), encoder.summary())
            print(
                "screen=%.5f logloss=%.5f stability=%.5f params=%d branches=%s"
                % (
                    screen_score,
                    architecture_log_loss,
                    stability["mean_absolute_probability_shift"],
                    effective_parameters,
                    encoder.summary()["selected_branches"],
                )
            )

        screen_frame = pd.DataFrame(screen_rows).sort_values(
            ["screen_score_lower_is_better", "effective_parameter_count", "name"]
        )
        screen_frame.to_csv(output / "architecture_screen.csv", index=False)
        top_names = set(
            screen_frame.head(min(settings["architecture_top_k"], len(screen_frame)))["name"].tolist()
        )
        finalists = [item for item in trained if item["config"]["name"] in top_names]

        por_rows = []
        resolved_arms = []
        for finalist_index, item in enumerate(finalists):
            config, encoder = item["config"], item["encoder"]
            print("\nFull POR finalist %d/%d: %s" % (finalist_index + 1, len(finalists), config["name"]))
            encoded, auxiliary = {}, {}
            for name, indices in (
                ("base", base),
                ("operator", operator),
                ("derivative", derivative_gate),
                ("wp_validation", wp_validation),
            ):
                result = encoder.encode_source(source, indices, batch_size=args.read_batch_size)
                encoded[name] = result["geometry"]
                auxiliary[name] = result["auxiliary_probability"]
            adapter = IdentityFeatureMap(config["geometry"])
            model = DerivativeGatedNeuralPOR(
                n_neighbors=args.neighbors,
                scale_neighbor=args.scale_neighbor,
                lambda_pi_grid=settings["lambda_pi_grid"],
                k_max=settings["k_max"],
                dimension_bootstrap_repetitions=settings["dimension_bootstrap_repetitions"],
                neural_max_epochs=settings["neural_max_epochs"],
                neural_patience=settings["neural_patience"],
                neural_architecture=config["h_theta"],
                measurement_rank=4,
                measurement_max_iter=settings["measurement_max_iter"],
                measurement_restarts=settings["measurement_restarts"],
                max_measurement_samples=settings["measurement_sample"],
                max_graph_samples=settings["graph_sample"],
                query_batch_size=args.query_batch_size,
                random_state=args.random_state + 5000,
            ).fit(
                encoded["base"],
                y[base],
                X_operator_validation=encoded["operator"],
                y_operator_validation=y[operator],
                X_dimension_validation=encoded["derivative"],
                y_dimension_validation=y[derivative_gate],
                X_reference_train=encoded["base"],
                X_reference_dimension=encoded["derivative"],
            )
            probability = model.predict_proba(encoded["wp_validation"])[:, 1]
            working_point = background_rejection_at_signal_efficiency(
                y[wp_validation], probability, 0.70
            )
            eligible = bool(model.criterion_reached_ and model.measurement_.optimization_result_.success)
            row = {
                "name": config["name"],
                "eligible_for_selection": eligible,
                "h_theta_architecture": config["h_theta"],
                "effective_parameter_count": encoder.parameter_count(effective=True),
                "selected_branches": ";".join(encoder.summary()["selected_branches"]),
                "candidate_K_PF": int(model.candidate_bank_k_),
                "K_DG": int(model.selected_k_),
                "selected_orbitals_one_based": ";".join(
                    str(int(value + 1)) for value in model.selected_orbital_indices_
                ),
                **working_point,
                "locked_test_used": False,
            }
            por_rows.append(row)
            resolved_arms.append(
                {
                    "config": config,
                    "encoder": encoder,
                    "adapter": adapter,
                    "model": model,
                    "working_point": working_point,
                    "eligible": eligible,
                }
            )
            print(
                "Rlight@70=%.3f [%.3f, %.3f], mistags=%d/%d, KDG=%d"
                % (
                    working_point["light_rejection"],
                    working_point["light_rejection_lower_68"],
                    working_point["light_rejection_upper_68"],
                    working_point["light_mistagged_jets"],
                    working_point["total_light_jets"],
                    model.selected_k_,
                )
            )
            del encoded, auxiliary
            gc.collect()

    pd.DataFrame(por_rows).to_csv(output / "development_por_finalists.csv", index=False)
    eligible_arms = [arm for arm in resolved_arms if arm["eligible"]]
    if not eligible_arms:
        save_json(output / "test_scoring_status.json", {"scored": False, "reason": "no resolved self-configuring POR finalist"})
        raise RuntimeError("No self-configuring POR finalist resolved.")
    selected = _parameter_tie_selection(eligible_arms)
    config, encoder, adapter, model = (
        selected["config"], selected["encoder"], selected["adapter"], selected["model"]
    )
    source_hash = workspace_source_sha256()

    _save_model_paths(model, output)
    _save_core_artifacts(model, rows[base], output, args)
    save_json(output / "role_split_audit.json", split_audit)
    save_json(output / "official_file_integrity.json", integrity)
    save_json(output / "jetset_schema_and_leakage_audit.json", dataset.schema_audit)
    save_json(output / "selected_self_configuring_encoder.json", encoder.summary())
    save_json(output / "selected_derivative_gated_dimension.json", model.selected_)
    save_json(output / "measurement_summary.json", measurement_summary(model))
    save_json(
        output / "selected_development_model.json",
        {
            "selection_rule": "max lower-68 Rlight@70; within 1% choose fewer effective encoder parameters",
            "selected_architecture": config,
            "selected_encoder": encoder.summary(),
            "working_point": selected["working_point"],
            "locked_test_used_for_selection": False,
        },
    )
    np.savez_compressed(
        output / "frozen_split_indices.npz",
        sampled_source_rows=rows,
        sampled_event_numbers=events,
        base=base,
        operator_validation=operator,
        dimension_gate=dimension,
        derivative_gate=derivative_gate,
        wp_validation=wp_validation,
        independent_test=locked_test,
        encoder_train=encoder_train,
        encoder_early_stop=encoder_early_stop,
        architecture_validation=architecture_validation,
    )
    bundle = {
        "structured_track_encoder": encoder,
        "input_feature_adapter": adapter,
        "neural_por_model": model,
        "selected_architecture": config,
        "selected_representation": "self-configuring-gated-hybrid",
        "dataset": "atlas-jetset-b-light-self-configuring-tracks",
        "dataset_tier": tier,
        "official_file_name": h5_file.name,
        "official_file_size": int(integrity["size_bytes"]),
        "official_file_adler32": integrity["adler32"],
        "label_definition": args.label_definition,
        "label_field": dataset.label_field,
        "analysis_protocol": args.protocol,
        "max_source_events": int(args.max_source_events),
        "sample": int(settings["sample"]),
        "random_state": int(args.random_state),
        "max_tracks": int(settings["max_tracks"]),
        "sampled_source_rows": rows,
        "sampled_event_numbers": events,
        "independent_test_indices": locked_test,
        "source_tree_sha256": source_hash,
        "method_resolved_before_test": True,
    }
    joblib.dump(bundle, output / "trained_self_configuring_dg_npor_v9.joblib", compress=3)
    save_json(
        output / "test_scoring_status.json",
        {
            "scored": False,
            "reason": "self-configuring architecture frozen; run score_self_configuring_locked_test.py once",
            "locked_test_track_tensor_read_during_scan": False,
            "test_truth_or_tagger_audit_used_for_selection": False,
        },
    )
    run_config = vars(args).copy()
    run_config.update(settings)
    run_config["legal_architecture_library"] = architecture_library(args.preset)
    save_json(output / "run_config.json", run_config)
    save_json(
        output / "reproducibility_manifest.json",
        {
            "dataset": dataset.dataset_name,
            "rows_used": int(len(y)),
            "unique_events_used": int(len(np.unique(events))),
            "selected_architecture": config,
            "selected_branches": encoder.summary()["selected_branches"],
            "official_file": integrity,
            "source_tree_sha256": source_hash,
            "python": platform.python_version(),
            "locked_test_scored": False,
        },
    )
    print("\nSelected self-configuring architecture:", json.dumps(config, sort_keys=True))
    print("Selected geometry branches:", encoder.summary()["selected_branches"])
    print("K_DG:", model.selected_k_)
    print("Selected orbitals (one-based):", [int(value + 1) for value in model.selected_orbital_indices_])
    print("Locked test remains closed.")
    print("Next: score_self_configuring_locked_test.py", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
