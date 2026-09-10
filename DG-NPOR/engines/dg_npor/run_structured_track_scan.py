#!/usr/bin/env python3
"""Development-only local/sequential/parallel/hybrid geometry scan.

All arms use the same selected jets, eventNumber-modulo-10 protocol, and model
seed. The residue-9 locked-test track tensor is never read here.
Representation choice uses only a separate eventwise split of residue 8 and
the lower 68% confidence bound of light rejection at 70% b efficiency.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples._v9_experiment_backend import index_sha256, measurement_summary, save_json
from por_core import DerivativeGatedNeuralPOR
from por_core.working_point_scan import eventwise_development_split, stratified_take
from por_hep.atlas_jetset import (
    download_official_file,
    infer_official_tier,
    verify_official_file,
)
from por_hep.flavour_metrics import background_rejection_at_signal_efficiency
from por_hep.structured_tracks import (
    H5StructuredTrackSource,
    IdentityFeatureMap,
    STRUCTURED_REPRESENTATIONS,
    StructuredTrackGeometry,
    load_atlas_jetset_index,
)
from run_atlas_jetset_por import (
    _role_audit,
    _save_core_artifacts,
    _save_model_paths,
    _split_roles,
    workspace_source_sha256,
)


def preset_defaults(preset):
    if preset == "quick":
        return {
            "sample": 20000,
            "encoder_sample": 20000,
            "max_tracks": 20,
            "encoder_hidden": 24,
            "geometry_dimension": 48,
            "encoder_epochs": 12,
            "encoder_patience": 3,
            "encoder_batch_size": 512,
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
        "sample": 100000,
        "encoder_sample": 60000,
        "max_tracks": 40,
        "encoder_hidden": 32,
        "geometry_dimension": 64,
        "encoder_epochs": 20,
        "encoder_patience": 5,
        "encoder_batch_size": 512,
        "graph_sample": 12000,
        "k_max": 128,
        "dimension_bootstrap_repetitions": 200,
        "neural_max_epochs": 180,
        "neural_patience": 20,
        "measurement_sample": 8000,
        "measurement_max_iter": 1200,
        "measurement_restarts": 2,
        "lambda_pi_grid": (0.05, 0.10, 0.25, 0.50, 1.0, 2.0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare local, sequential, parallel, and hybrid track geometry POR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--protocol", choices=("pilot-b-light", "paper-ttbar"), default="paper-ttbar")
    parser.add_argument("--max-source-events", type=int, default=2000000)
    parser.add_argument("--dataset-tier", choices=("small", "medium", "large"), default="small")
    parser.add_argument(
        "--h5-file", default="data/atlas_jetset/mc-flavtag-ttbar-small.h5"
    )
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--label-definition", choices=("cone", "ghost"), default="cone")
    parser.add_argument(
        "--representations", default="graph,sequential,parallel,hybrid"
    )
    parser.add_argument("--encoder-sample", type=int, default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--encoder-hidden", type=int, default=None)
    parser.add_argument("--geometry-dimension", type=int, default=None)
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--encoder-patience", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-3)
    parser.add_argument("--graph-track-neighbors", type=int, default=6)
    parser.add_argument("--sequential-steps", type=int, default=3)
    parser.add_argument("--parallel-temperature", type=float, default=2.0)
    parser.add_argument("--parallel-floor", type=float, default=0.02)
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
    parser.add_argument(
        "--output-dir", default="outputs/atlas_jetset_multiscale_geometry_scan"
    )
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _settings(args):
    defaults = preset_defaults(args.preset)
    names = (
        "sample",
        "encoder_sample",
        "max_tracks",
        "encoder_hidden",
        "geometry_dimension",
        "encoder_epochs",
        "encoder_patience",
        "encoder_batch_size",
        "graph_sample",
        "k_max",
        "dimension_bootstrap_repetitions",
        "neural_max_epochs",
        "neural_patience",
        "measurement_sample",
        "measurement_max_iter",
        "measurement_restarts",
    )
    values = {
        name: defaults[name] if getattr(args, name) is None else int(getattr(args, name))
        for name in names
    }
    values["lambda_pi_grid"] = defaults["lambda_pi_grid"]
    return values


def _parse_representations(value):
    values = tuple(dict.fromkeys(token.strip() for token in str(value).split(",") if token.strip()))
    if not values or any(value not in STRUCTURED_REPRESENTATIONS for value in values):
        raise ValueError(
            "--representations must contain set, graph, sequential, parallel, and/or hybrid."
        )
    return values


def _eventwise_encoder_split(pool, groups, labels, random_state, validation_fraction=0.15):
    pool = np.asarray(pool, dtype=int)
    groups = np.asarray(groups, dtype=np.int64)
    unique = np.unique(groups[pool]).astype(np.uint64)
    with np.errstate(over="ignore"):
        values = unique + np.uint64(0x9E3779B97F4A7C15 + int(random_state))
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    threshold = int(round(10000 * float(validation_fraction)))
    validation_events = unique[(values % np.uint64(10000)) < np.uint64(threshold)]
    validation_mask = np.isin(groups[pool].astype(np.uint64), validation_events)
    train = pool[~validation_mask]
    validation = pool[validation_mask]
    if (
        len(train) == 0
        or len(validation) == 0
        or len(np.unique(labels[train])) != 2
        or len(np.unique(labels[validation])) != 2
    ):
        train, validation = eventwise_development_split(
            pool, groups, random_state=random_state
        )
    if len(np.intersect1d(groups[train], groups[validation])):
        raise RuntimeError("Event leakage in structured encoder early stopping split.")
    return train.astype(int), validation.astype(int)


def _arm_key(arm):
    row = arm["working_point"]
    return (
        -float(row["light_rejection_lower_68"]),
        -float(row["light_rejection"]),
        int(arm["model"].selected_k_),
        str(arm["representation"]),
    )


def main(argv=None):
    args = parse_args(argv)
    settings = _settings(args)
    representations = _parse_representations(args.representations)
    if args.scale_neighbor > args.neighbors:
        raise ValueError("scale-neighbor cannot exceed neighbors.")
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
    y = dataset.y
    events = dataset.group_id
    rows = dataset.row_index
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
        base,
        y,
        min(settings["encoder_sample"], len(base)),
        args.random_state + 601,
    )
    encoder_train, encoder_validation = _eventwise_encoder_split(
        encoder_pool, events, y, args.random_state + 611
    )
    comparison_seed = int(args.random_state) + 1000

    print("Dataset:", dataset.dataset_name)
    print("Selected jets:", len(y), "with up to", settings["max_tracks"], "tracks per jet")
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
    print(
        "Roles: base=%d operator=%d derivative=%d WP-validation=%d locked-test=%d"
        % (len(base), len(operator), len(derivative_gate), len(wp_validation), len(locked_test))
    )
    print("Encoder fit/early-stop rows:", len(encoder_train), len(encoder_validation))
    print("Common representation-comparison seed:", comparison_seed)
    print("Locked-test track tensors and POR predictions are not opened by this script.")
    print("Test truth/tagger audit records are not used for model or representation selection.")
    print("GN2/DL1 and GN2 vertex assignments are never passed to POR.")

    arm_results = []
    development_rows = []
    with H5StructuredTrackSource(
        h5_file, rows, max_tracks=settings["max_tracks"]
    ) as source:
        for arm_index, representation in enumerate(representations):
            print("\nStructured arm %d/%d: %s-POR" % (arm_index + 1, len(representations), representation))
            encoder = StructuredTrackGeometry(
                representation=representation,
                hidden_dimension=settings["encoder_hidden"],
                geometry_dimension=settings["geometry_dimension"],
                max_tracks=settings["max_tracks"],
                graph_neighbors=args.graph_track_neighbors,
                sequential_steps=args.sequential_steps,
                parallel_temperature=args.parallel_temperature,
                parallel_floor=args.parallel_floor,
                learning_rate=args.encoder_learning_rate,
                batch_size=settings["encoder_batch_size"],
                max_epochs=settings["encoder_epochs"],
                patience=settings["encoder_patience"],
                random_state=comparison_seed,
            ).fit(
                source,
                encoder_train,
                y[encoder_train],
                encoder_validation,
                y[encoder_validation],
                read_batch_size=max(args.read_batch_size, 2048),
            )
            pd.DataFrame(encoder.history_).to_csv(
                output / ("%s_encoder_training.csv" % representation), index=False
            )
            save_json(
                output / ("%s_encoder_summary.json" % representation),
                encoder.summary(),
            )

            encoded = {}
            auxiliary = {}
            for name, indices in (
                ("base", base),
                ("operator", operator),
                ("derivative", derivative_gate),
                ("wp_validation", wp_validation),
            ):
                result = encoder.encode_source(
                    source, indices, batch_size=args.read_batch_size
                )
                encoded[name] = result["geometry"]
                auxiliary[name] = result["auxiliary_probability"]

            adapter = IdentityFeatureMap(settings["geometry_dimension"])
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
                random_state=comparison_seed,
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
            auxiliary_wp = background_rejection_at_signal_efficiency(
                y[wp_validation], auxiliary["wp_validation"], 0.70
            )
            eligible = bool(
                model.criterion_reached_
                and model.measurement_.optimization_result_.success
            )
            row = {
                "representation": representation,
                "eligible_for_selection": eligible,
                "max_tracks": settings["max_tracks"],
                "structured_geometry_dimension": settings["geometry_dimension"],
                "common_comparison_seed": comparison_seed,
                "sequential_steps": (
                    args.sequential_steps
                    if representation in {"sequential", "hybrid"}
                    else 0
                ),
                "dense_parallel_broadcast": bool(
                    representation in {"parallel", "hybrid"}
                ),
                "parallel_temperature": (
                    args.parallel_temperature
                    if representation in {"parallel", "hybrid"}
                    else None
                ),
                "parallel_floor": (
                    args.parallel_floor
                    if representation in {"parallel", "hybrid"}
                    else None
                ),
                "candidate_K_PF": int(model.candidate_bank_k_),
                "K_DG": int(model.selected_k_),
                "selected_orbitals_one_based": ";".join(
                    str(int(value + 1)) for value in model.selected_orbital_indices_
                ),
                **working_point,
                "auxiliary_Rlight_at_70_audit_only": float(
                    auxiliary_wp["light_rejection"]
                ),
                "auxiliary_probability_used_by_POR": False,
            }
            development_rows.append(row)
            arm_results.append(
                {
                    "representation": representation,
                    "encoder": encoder,
                    "adapter": adapter,
                    "model": model,
                    "working_point": working_point,
                    "eligible": eligible,
                }
            )
            print(
                "%s-POR development: Rlight@70=%.3f [%0.3f, %0.3f], mistags=%d/%d, KDG=%d"
                % (
                    representation,
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

    pd.DataFrame(development_rows).to_csv(
        output / "development_representation_comparison.csv", index=False
    )
    eligible_arms = [arm for arm in arm_results if arm["eligible"]]
    if not eligible_arms:
        save_json(
            output / "test_scoring_status.json",
            {"scored": False, "reason": "no resolved structured POR arm"},
        )
        raise RuntimeError("Neither structured representation produced a resolved POR model.")
    selected = min(eligible_arms, key=_arm_key)
    model = selected["model"]
    encoder = selected["encoder"]
    adapter = selected["adapter"]
    selected_representation = selected["representation"]
    source_hash = workspace_source_sha256()

    _save_model_paths(model, output)
    _save_core_artifacts(model, rows[base], output, args)
    save_json(output / "role_split_audit.json", split_audit)
    save_json(output / "official_file_integrity.json", integrity)
    save_json(output / "jetset_schema_and_leakage_audit.json", dataset.schema_audit)
    save_json(output / "structured_track_encoder.json", encoder.summary())
    save_json(output / "selected_derivative_gated_dimension.json", model.selected_)
    save_json(output / "measurement_summary.json", measurement_summary(model))
    save_json(
        output / "selected_development_model.json",
        {
            "selection_metric": "maximum lower 68% Clopper-Pearson bound of Rlight at 70% b efficiency",
            "selected_representation": selected_representation,
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
        encoder_validation=encoder_validation,
    )
    bundle = {
        "structured_track_encoder": encoder,
        "input_feature_adapter": adapter,
        "neural_por_model": model,
        "selected_representation": selected_representation,
        "dataset": "atlas-jetset-b-light-structured-tracks",
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
        "model_comparison_seed": comparison_seed,
        "max_tracks": int(settings["max_tracks"]),
        "sampled_source_rows": rows,
        "sampled_event_numbers": events,
        "independent_test_indices": locked_test,
        "source_tree_sha256": source_hash,
        "method_resolved_before_test": True,
    }
    joblib.dump(bundle, output / "trained_structured_track_dg_npor_v9.joblib", compress=3)
    save_json(
        output / "test_scoring_status.json",
        {
            "scored": False,
            "reason": "representation frozen after development scan; run score_structured_locked_test.py once",
            "locked_test_track_tensor_read_during_scan": False,
            "locked_test_prediction_computed_during_scan": False,
            "test_truth_or_tagger_audit_used_for_selection": False,
        },
    )
    save_json(
        output / "method_status.json",
        {
            "method_resolved_before_test": True,
            "selected_representation": selected_representation,
            "structured_track_identity_preserved": True,
            "fixed_track_graph": bool(selected_representation != "set"),
            "sequential_local_propagation": bool(
                selected_representation in {"sequential", "hybrid"}
            ),
            "dense_parallel_broadcast": bool(
                selected_representation in {"parallel", "hybrid"}
            ),
            "original_track_residual_channel_preserved": True,
            "truth_track_fields_used": False,
            "GN2_or_DL1_scores_used_as_input": False,
            "GN2v01_vertexIndex_used": False,
            "auxiliary_probability_used_in_final_prediction": False,
            "operator": "H=L_sym+lambda_pi*V_PI",
            "K_DG": int(model.selected_k_),
            "selected_orbitals_one_based": [
                int(value + 1) for value in model.selected_orbital_indices_
            ],
            "test_scoring_deferred": True,
        },
    )
    run_config = vars(args).copy()
    run_config.update(settings)
    run_config["representations"] = list(representations)
    save_json(output / "run_config.json", run_config)
    save_json(
        output / "reproducibility_manifest.json",
        {
            "dataset": dataset.dataset_name,
            "rows_used": int(len(y)),
            "unique_events_used": int(len(np.unique(events))),
            "selected_representation": selected_representation,
            "development_comparison": development_rows,
            "official_file": integrity,
            "source_tree_sha256": source_hash,
            "python": platform.python_version(),
            "locked_test_scored": False,
        },
    )
    print("\nSelected development representation:", selected_representation)
    print("K_DG:", model.selected_k_)
    print(
        "Selected orbitals (one-based):",
        [int(value + 1) for value in model.selected_orbital_indices_],
    )
    print("Locked test remains closed.")
    print("Next: score_structured_locked_test.py", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
