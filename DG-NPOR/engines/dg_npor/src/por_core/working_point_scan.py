"""Development-only working-point scans for a frozen DG-NPOR basis.

This module never opens the independent test split.  It reuses a previously
fitted neural geometry, graph, predictive-information operator, and orbital
basis.  Candidate orbital banks and sparse derivative-ranked subsets are
compared on a separate working-point validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivative_gated_dimension import select_derivative_gated_dimension
from .orbitals import normalized_selected_orbital_state
from .readout import StructuredOrbitalMeasurement


def parse_positive_int_grid(value):
    """Parse a comma-separated, strictly increasing integer grid."""
    values = tuple(sorted(set(int(token.strip()) for token in str(value).split(","))))
    if not values or values[0] < 2:
        raise ValueError("Orbital grids must contain integers greater than or equal to two.")
    return values


def eventwise_development_split(indices, event_numbers, random_state=42):
    """Split complete events into derivative-gate and WP-validation halves.

    A SplitMix64 hash is used rather than event-number parity.  The latter is
    unsuitable because the ATLAS modulo-10 role already fixes the last digit.
    """
    indices = np.asarray(indices, dtype=int)
    event_numbers = np.asarray(event_numbers, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot split an empty development role.")
    unique_events = np.unique(event_numbers[indices]).astype(np.uint64)
    with np.errstate(over="ignore"):
        values = unique_events + np.uint64(0x9E3779B97F4A7C15 + int(random_state))
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    gate_events = unique_events[(values & np.uint64(1)) == 0]
    if len(gate_events) == 0 or len(gate_events) == len(unique_events):
        ordered = np.sort(unique_events)
        gate_events = ordered[::2]
    gate_mask = np.isin(event_numbers[indices].astype(np.uint64), gate_events)
    gate = indices[gate_mask]
    working_point = indices[~gate_mask]
    if len(gate) == 0 or len(working_point) == 0:
        raise RuntimeError("Eventwise development split produced an empty role.")
    overlap = np.intersect1d(event_numbers[gate], event_numbers[working_point])
    if len(overlap):
        raise RuntimeError("Event leakage between derivative gate and WP validation.")
    return gate.astype(int), working_point.astype(int)


def stratified_take(indices, labels, size, random_state):
    """Take a deterministic approximately class-balanced subset."""
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels, dtype=int)
    size = min(max(0, int(size)), len(indices))
    if size == len(indices):
        return indices.copy()
    if size == 0:
        return np.empty(0, dtype=int)
    rng = np.random.RandomState(int(random_state))
    classes = np.unique(labels[indices])
    if len(classes) != 2:
        raise ValueError("Stratified subset requires both binary classes.")
    per_class = [size // 2, size - size // 2]
    chosen = []
    for class_index, label in enumerate(classes):
        members = indices[labels[indices] == label]
        count = min(per_class[class_index], len(members))
        chosen.append(rng.choice(members, size=count, replace=False))
    selected = np.concatenate(chosen)
    if len(selected) < size:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        selected = np.concatenate(
            [selected, rng.choice(remaining, size=size - len(selected), replace=False)]
        )
    rng.shuffle(selected)
    return selected.astype(int)


def extend_frozen_orbitals(model, geometry, width):
    """Extend a frozen operator in batches and return its common stable prefix."""
    geometry = np.asarray(geometry, dtype=float)
    width = int(width)
    chunks = []
    common_stable = np.ones(width, dtype=bool)
    for block in model._iter_chunks(geometry):
        phi, stable = model._extend(model.core_, block, width)
        chunks.append(phi)
        common_stable &= np.asarray(stable, dtype=bool)
    raw = np.vstack(chunks) if chunks else np.empty((0, width), dtype=float)
    stable_prefix = model._stable_prefix(common_stable)
    return raw, int(stable_prefix)


def derivative_result_for_bank(model, phi_gate, bank_width, screening_rho=None):
    """Compute dJ/dgate ranking inside one fixed low-frequency bank."""
    width = int(bank_width)
    return select_derivative_gated_dimension(
        phi_train=model.core_.eigensystem.phi[:, :width],
        phi_validation=np.asarray(phi_gate, dtype=float)[:, :width],
        eigenvalues=model.core_.eigensystem.eigenvalues[:width],
        y_train=model.core_.base.y,
        priors=model.core_.base.priors,
        checkpoints=(width,),
        screening_rho=(
            float(model.screening_rho)
            if screening_rho is None
            else float(screening_rho)
        ),
        k_min=2,
        bootstrap_repetitions=0,
        random_state=int(model.random_state) + 9307 + width,
        candidate_bank_resolved=True,
    )


def degeneracy_safe_ranked_subset(dimension_result, requested_k):
    """Take derivative-ranked eigenspace blocks until requested K is reached."""
    requested_k = int(requested_k)
    block_id = np.asarray(dimension_result.block_id, dtype=int)
    selected = set()
    seen_blocks = set()
    for index in np.asarray(dimension_result.ranked_indices, dtype=int):
        block = int(block_id[index])
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        selected.update(np.flatnonzero(block_id == block).tolist())
        if len(selected) >= requested_k:
            break
    values = np.asarray(sorted(selected), dtype=int)
    if len(values) < 2:
        raise RuntimeError("Derivative ranking did not produce two orbitals.")
    return values


@dataclass
class ReadoutFitResult:
    measurement: StructuredOrbitalMeasurement
    training_path: tuple


def fit_structured_readout(
    train_state,
    train_labels,
    mining_state,
    mining_labels,
    *,
    rank=4,
    rho=1e-3,
    l2=1e-4,
    max_iter=800,
    restarts=1,
    hard_negative_iterations=0,
    hard_negative_fraction=0.10,
    hard_negative_weight=3.0,
    random_state=42,
):
    """Fit a PSD readout, optionally repeating development-only hard mining."""
    train_state = np.asarray(train_state, dtype=float)
    train_labels = np.asarray(train_labels, dtype=int)
    mining_state = np.asarray(mining_state, dtype=float)
    mining_labels = np.asarray(mining_labels, dtype=int)
    if set(np.unique(train_labels).tolist()) != {0, 1}:
        raise ValueError("Readout training requires both classes.")
    if len(mining_state) and set(np.unique(mining_labels).tolist()) != {0, 1}:
        raise ValueError("Hard-negative mining requires both classes.")

    def new_measurement(seed):
        return StructuredOrbitalMeasurement(
            rank=int(rank),
            rho=float(rho),
            l2=float(l2),
            max_iter=int(max_iter),
            n_restarts=int(restarts),
            random_state=int(seed),
        )

    measurement = new_measurement(random_state).fit(train_state, train_labels)
    path = [{
        "iteration": 0,
        "training_rows": int(len(train_labels)),
        "hard_light_rows": 0,
        "hard_b_rows": 0,
        "hard_row_weight": 1.0,
        "optimizer_success": bool(measurement.optimization_result_.success),
        "train_objective": float(measurement.train_objective_),
    }]
    if int(hard_negative_iterations) <= 0 or len(mining_state) == 0:
        return ReadoutFitResult(measurement, tuple(path))

    light = np.flatnonzero(mining_labels == 0)
    bottom = np.flatnonzero(mining_labels == 1)
    hard_count = max(
        1,
        int(np.ceil(float(hard_negative_fraction) * min(len(light), len(bottom)))),
    )
    hard_count = min(hard_count, len(light), len(bottom))
    for iteration in range(1, int(hard_negative_iterations) + 1):
        probability = measurement.predict_proba(mining_state)[:, 1]
        hard_light = light[np.argsort(probability[light])[-hard_count:]]
        hard_bottom = bottom[np.argsort(probability[bottom])[:hard_count]]
        hard = np.concatenate([hard_light, hard_bottom])
        augmented_state = np.vstack([train_state, mining_state[hard]])
        augmented_labels = np.concatenate([train_labels, mining_labels[hard]])
        weights = np.concatenate([
            np.ones(len(train_labels), dtype=float),
            np.full(len(hard), float(hard_negative_weight), dtype=float),
        ])
        measurement = new_measurement(
            int(random_state) + 1009 * iteration
        ).fit(augmented_state, augmented_labels, sample_weight=weights)
        path.append({
            "iteration": int(iteration),
            "training_rows": int(len(augmented_labels)),
            "hard_light_rows": int(len(hard_light)),
            "hard_b_rows": int(len(hard_bottom)),
            "hard_row_weight": float(hard_negative_weight),
            "minimum_selected_light_score": float(np.min(probability[hard_light])),
            "maximum_selected_b_score": float(np.max(probability[hard_bottom])),
            "optimizer_success": bool(measurement.optimization_result_.success),
            "train_objective": float(measurement.train_objective_),
        })
    return ReadoutFitResult(measurement, tuple(path))


def selected_state(raw_orbitals, selected_indices):
    return normalized_selected_orbital_state(
        np.asarray(raw_orbitals, dtype=float),
        np.asarray(selected_indices, dtype=int),
    )
