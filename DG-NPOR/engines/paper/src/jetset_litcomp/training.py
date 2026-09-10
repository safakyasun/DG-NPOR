"""Training loop for a same-split ParticleNet JetSet baseline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .data import (
    COORDINATE_INDICES,
    JetSetBatchReader,
    RobustTrackScaler,
    load_frozen_split,
)
from .particle_net import ParticleNet, require_torch, select_device, torch


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    patience: int
    batch_size: int
    initial_lr: float
    peak_lr: float
    final_lr: float


def settings_for(architecture, preset):
    architecture = str(architecture)
    preset = str(preset)
    if architecture not in {"full", "lite"}:
        raise ValueError("architecture must be full or lite.")
    if preset not in {"quick", "paper"}:
        raise ValueError("preset must be quick or paper.")
    full = architecture == "full"
    if preset == "paper":
        return TrainingSettings(
            epochs=20,
            patience=20,
            batch_size=256 if full else 512,
            initial_lr=3e-4 if full else 5e-4,
            peak_lr=3e-3 if full else 5e-3,
            final_lr=5e-7 if full else 1e-6,
        )
    return TrainingSettings(
        epochs=8,
        patience=4,
        batch_size=256 if full else 512,
        initial_lr=3e-4 if full else 5e-4,
        peak_lr=3e-3 if full else 5e-3,
        final_lr=5e-7 if full else 1e-6,
    )


def _learning_rate(epoch, epochs, initial, peak, final):
    """Piecewise 40% warmup, 40% decay, 20% cooldown schedule."""
    position = float(epoch) / max(1, int(epochs) - 1)
    if position <= 0.40:
        return float(initial + (peak - initial) * position / 0.40)
    if position <= 0.80:
        return float(peak + (initial - peak) * (position - 0.40) / 0.40)
    return float(initial + (final - initial) * (position - 0.80) / 0.20)


def _set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_batch(reader, scaler, logical_indices, labels, device):
    tracks, mask, context = reader.read(logical_indices)
    tracks, mask, context = scaler.transform(tracks, mask, context)
    return (
        torch.from_numpy(tracks).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(context).to(device),
        torch.from_numpy(labels[np.asarray(logical_indices, dtype=int)].astype(np.float32)).to(
            device
        ),
    )


def predict_probabilities(
    model,
    reader,
    scaler,
    logical_indices,
    labels,
    device,
    batch_size,
):
    model.eval()
    logical_indices = np.asarray(logical_indices, dtype=int)
    values = []
    with torch.no_grad():
        for start in range(0, len(logical_indices), int(batch_size)):
            batch = logical_indices[start : start + int(batch_size)]
            tracks, mask, context, _ = _tensor_batch(
                reader, scaler, batch, labels, device
            )
            logits = model(tracks, mask, context)
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values).astype(float)


def _evaluate(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy_at_0_5": float(accuracy_score(y_true, probability >= 0.5)),
        "log_loss": float(log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7))),
    }


def _balanced_scaler_sample(train_indices, labels, maximum, seed):
    train_indices = np.asarray(train_indices, dtype=int)
    maximum = min(int(maximum), len(train_indices))
    if maximum <= 0:
        raise ValueError("scaler_sample must be positive.")
    if maximum == len(train_indices):
        return np.sort(train_indices)
    rng = np.random.RandomState(int(seed) + 991)
    per_class = maximum // 2
    chosen = []
    for label in (0, 1):
        candidates = train_indices[labels[train_indices] == label]
        take = min(per_class, len(candidates))
        chosen.append(rng.choice(candidates, size=take, replace=False))
    remainder = maximum - sum(len(value) for value in chosen)
    if remainder:
        already = np.concatenate(chosen)
        candidates = np.setdiff1d(train_indices, already, assume_unique=False)
        chosen.append(rng.choice(candidates, size=remainder, replace=False))
    return np.sort(np.concatenate(chosen).astype(int))


def train_particle_net(
    selfconfig_output_dir,
    h5_file,
    output_dir,
    architecture="full",
    preset="quick",
    device="auto",
    seed=42,
    batch_size=None,
    epochs=None,
    patience=None,
    scaler_sample=50000,
    max_tracks=20,
    weight_decay=1e-4,
):
    """Train on frozen development roles and score the locked test once."""
    require_torch()
    split = load_frozen_split(selfconfig_output_dir)
    settings = settings_for(architecture, preset)
    batch_size = int(batch_size or settings.batch_size)
    epochs = int(epochs or settings.epochs)
    patience = int(patience or settings.patience)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h5_file = Path(h5_file)
    if not h5_file.exists():
        raise FileNotFoundError(
            f"Official HDF5 file not found: {h5_file}. This workspace never downloads it."
        )

    selected_device = select_device(device)
    _set_seed(seed)
    train_indices = split.particle_net_train
    validation_indices = np.asarray(split.wp_validation, dtype=int)
    test_indices = np.asarray(split.independent_test, dtype=int)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("ParticleNet train and validation overlap.")
    if np.intersect1d(
        np.concatenate([train_indices, validation_indices]), test_indices
    ).size:
        raise RuntimeError("ParticleNet development roles overlap locked test.")

    print(f"Device: {selected_device}")
    print(
        "Frozen roles: ParticleNet train=%d validation=%d locked-test=%d"
        % (len(train_indices), len(validation_indices), len(test_indices))
    )
    print("The locked test is not used for epoch or architecture selection.")

    with JetSetBatchReader(h5_file, split.source_rows, max_tracks=max_tracks) as reader:
        labels, events = reader.read_labels_and_events()
        if not np.array_equal(events, split.event_numbers):
            raise RuntimeError("Official HDF5 event numbers differ from frozen split.")

        scale_indices = _balanced_scaler_sample(
            train_indices, labels, scaler_sample, seed
        )
        scale_tracks, scale_mask, scale_context = reader.read(scale_indices)
        scaler = RobustTrackScaler().fit(scale_tracks, scale_mask, scale_context)
        scaler.save(output / "particle_net_input_scaler.npz")
        del scale_tracks, scale_mask, scale_context

        model = ParticleNet(
            input_features=19,
            context_features=4,
            coordinate_indices=COORDINATE_INDICES,
            architecture=architecture,
        ).to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.initial_lr, weight_decay=float(weight_decay)
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        rng = np.random.RandomState(int(seed) + 17)
        best_state = None
        best_epoch = None
        best_key = None
        stale = 0
        history = []

        for epoch in range(epochs):
            started = time.time()
            lr = _learning_rate(
                epoch,
                epochs,
                settings.initial_lr,
                settings.peak_lr,
                settings.final_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            shuffled = train_indices.copy()
            rng.shuffle(shuffled)
            model.train()
            losses = []
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                if len(batch) < 2:
                    continue
                tracks, mask, context, target = _tensor_batch(
                    reader, scaler, batch, labels, selected_device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(tracks, mask, context)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            validation_probability = predict_probabilities(
                model,
                reader,
                scaler,
                validation_indices,
                labels,
                selected_device,
                batch_size,
            )
            validation = _evaluate(labels[validation_indices], validation_probability)
            row = {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train_batch_loss": float(np.mean(losses)),
                "validation_auc": validation["auc"],
                "validation_accuracy_at_0_5": validation["accuracy_at_0_5"],
                "validation_log_loss": validation["log_loss"],
                "seconds": float(time.time() - started),
            }
            history.append(row)
            print(
                "Epoch %02d/%02d: loss=%.5f val_acc=%.5f val_auc=%.5f val_logloss=%.5f"
                % (
                    epoch + 1,
                    epochs,
                    row["train_batch_loss"],
                    row["validation_accuracy_at_0_5"],
                    row["validation_auc"],
                    row["validation_log_loss"],
                )
            )
            # Original ParticleNet chose epochs by validation accuracy. AUC and
            # log loss only break exact numerical ties; the locked test is closed.
            key = (
                row["validation_accuracy_at_0_5"],
                row["validation_auc"],
                -row["validation_log_loss"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if preset == "quick" and stale >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        if best_state is None:
            raise RuntimeError("ParticleNet training did not produce a checkpoint.")
        model.load_state_dict(best_state)
        torch.save(
            {
                "state_dict": best_state,
                "architecture": architecture,
                "model_summary": model.summary(),
                "best_epoch": int(best_epoch),
                "seed": int(seed),
            },
            output / "particle_net_selected_model.pt",
        )
        pd.DataFrame(history).to_csv(output / "particle_net_training_history.csv", index=False)

        selected_validation_probability = predict_probabilities(
            model,
            reader,
            scaler,
            validation_indices,
            labels,
            selected_device,
            batch_size,
        )
        selected_validation_metrics = _evaluate(
            labels[validation_indices], selected_validation_probability
        )
        pd.DataFrame(
            {
                "source_row": split.source_rows[validation_indices],
                "event_number": split.event_numbers[validation_indices],
                "y_true_light0_b1": labels[validation_indices],
                "particle_net_probability_b": selected_validation_probability,
            }
        ).to_csv(
            output / "particle_net_wp_validation_predictions.csv.gz",
            index=False,
        )

        print("Development selection is frozen. Opening the locked test once.")
        test_probability = predict_probabilities(
            model,
            reader,
            scaler,
            test_indices,
            labels,
            selected_device,
            batch_size,
        )
        test_metrics = _evaluate(labels[test_indices], test_probability)
        pd.DataFrame(
            {
                "source_row": split.source_rows[test_indices],
                "event_number": split.event_numbers[test_indices],
                "y_true_light0_b1": labels[test_indices],
                "particle_net_probability_b": test_probability,
            }
        ).to_csv(output / "particle_net_locked_test_predictions.csv.gz", index=False)

    manifest = {
        "inputs": {"max_tracks": int(max_tracks), "track_features": 19, "jet_context_features": 4},
        "method": f"{model.config.name}-JetSet-retrained",
        "architecture": architecture,
        "model": model.summary(),
        "paper_architecture_reimplemented": True,
        "published_particle_net_test_number_reused": False,
        "input_domain": "ATLAS JetSet reconstructed tracks, not original ParticleNet constituents",
        "truth_gn2_dl1_used_as_input": False,
        "source_selfconfig_output": str(Path(selfconfig_output_dir).resolve()),
        "source_h5_file": str(h5_file.resolve()),
        "frozen_roles": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "locked_test": int(len(test_indices)),
        },
        "selection": {
            "criterion": "maximum validation accuracy; AUC/log-loss tie breakers",
            "best_epoch": int(best_epoch),
            "locked_test_used_for_selection": False,
            "selected_checkpoint_validation_metrics": selected_validation_metrics,
        },
        "training": {
            "preset": preset,
            "requested_epochs": epochs,
            "completed_epochs": len(history),
            "batch_size": batch_size,
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "device": str(selected_device),
            "scaler_fit_rows": int(len(scale_indices)),
        },
        "locked_test_metrics_diagnostic": test_metrics,
    }
    with (output / "particle_net_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
