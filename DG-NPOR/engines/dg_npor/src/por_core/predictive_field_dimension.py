"""Automatic predictive-field orbital dimension.

This is an explicitly named Neural-POR extension, not PDF Section 1.13.  It
preserves the PDF operator and ordered orbital prefix, but infers dimension
from the unpenalized spectral support of the class-direction field.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .risk_audit import fit_point_losses, make_probabilistic_probe
from .orbitals import (
    normalized_orbital_state,
    normalized_selected_orbital_state,
)


EPS = 1e-12


@dataclass(frozen=True)
class PredictiveFieldDimensionResult:
    selected_k: int
    participation_rank: float
    entropy_rank: float
    mass_location_k: int
    cutoff_block_end: int
    captured_weight: float
    field_weighted_mean: float
    coefficients: np.ndarray
    coefficient_power: np.ndarray
    predictive_weights: np.ndarray
    normalized_weights: np.ndarray
    cumulative_weights: np.ndarray
    block_id: np.ndarray
    bootstrap_k: np.ndarray
    checkpoint_records: tuple
    spectrum_stable: bool

    def summary(self):
        if len(self.bootstrap_k):
            values, counts = np.unique(self.bootstrap_k, return_counts=True)
            bootstrap_mode = int(values[np.argmax(counts)])
            bootstrap_mode_fraction = float(np.max(counts) / len(self.bootstrap_k))
            bootstrap_exact_fraction = float(
                np.mean(self.bootstrap_k == self.selected_k)
            )
        else:
            bootstrap_mode = None
            bootstrap_mode_fraction = None
            bootstrap_exact_fraction = None
        return {
            "selection_method": "predictive_field_spectral_support",
            "user_selected_K": False,
            "user_selected_delta": False,
            "K_PF": int(self.selected_k),
            "participation_rank": float(self.participation_rank),
            "entropy_rank_diagnostic": float(self.entropy_rank),
            "mass_location_K": int(self.mass_location_k),
            "degeneracy_safe_cutoff": int(self.cutoff_block_end),
            "captured_predictive_field_power": float(self.captured_weight),
            "extra_eigenvalue_penalty": False,
            "spectrum_checkpoint_stable": bool(self.spectrum_stable),
            "bootstrap_repetitions": int(len(self.bootstrap_k)),
            "bootstrap_K_median": (
                None if not len(self.bootstrap_k)
                else float(np.median(self.bootstrap_k))
            ),
            "bootstrap_K_min": (
                None if not len(self.bootstrap_k) else int(np.min(self.bootstrap_k))
            ),
            "bootstrap_K_max": (
                None if not len(self.bootstrap_k) else int(np.max(self.bootstrap_k))
            ),
            "bootstrap_K_mode": bootstrap_mode,
            "bootstrap_K_mode_fraction": bootstrap_mode_fraction,
            "bootstrap_exact_selected_K_fraction": bootstrap_exact_fraction,
        }

    def records(self, eigenvalues):
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        return [
            {
                "orbital": int(k + 1),
                "eigenvalue": float(eigenvalues[k]),
                "block_id": int(self.block_id[k]),
                "field_coefficient": float(self.coefficients[k]),
                "coefficient_power": float(self.coefficient_power[k]),
                "predictive_field_power": float(self.predictive_weights[k]),
                "normalized_weight": float(self.normalized_weights[k]),
                "cumulative_weight": float(self.cumulative_weights[k]),
                "selected": bool(k < self.selected_k),
            }
            for k in range(len(eigenvalues))
        ]


def prior_corrected_log_odds_field(posterior, priors):
    posterior = np.asarray(posterior, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if posterior.ndim != 2 or posterior.shape[1] != 2 or len(priors) != 2:
        raise ValueError("The current predictive field is binary-class only.")
    return (
        np.log(np.maximum(posterior[:, 1], EPS) / max(priors[1], EPS))
        - np.log(np.maximum(posterior[:, 0], EPS) / max(priors[0], EPS))
    )


def _spectral_blocks(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)
    tolerance = 100.0 * np.sqrt(np.finfo(float).eps) * max(
        1.0, float(np.max(np.abs(values)))
    )
    block_id = np.zeros(len(values), dtype=int)
    block = 0
    for index in range(1, len(values)):
        if values[index] - values[index - 1] > tolerance:
            block += 1
        block_id[index] = block
    return block_id


def _select_prefix(normalized_weights, block_id, k_min=2):
    q = np.asarray(normalized_weights, dtype=float)
    participation = float(1.0 / max(np.sum(q * q), EPS))
    effective_count = int(np.ceil(participation - 1e-12))
    target_mass = max(0.0, 1.0 - 1.0 / max(participation, 1.0))
    cumulative = np.cumsum(q)
    mass_k = int(np.searchsorted(cumulative, target_mass, side="left") + 1)
    selected = min(len(q), max(int(k_min), effective_count, mass_k))
    cutoff_block = block_id[selected - 1]
    block_end = int(np.flatnonzero(block_id == cutoff_block)[-1] + 1)
    selected = min(len(q), max(selected, block_end))
    return selected, participation, mass_k, cumulative


def _weights_from_row_multiplicity(phi, field, degree, block_id,
                                   multiplicity=None):
    phi = np.asarray(phi, dtype=float)
    field = np.asarray(field, dtype=float)
    degree = np.asarray(degree, dtype=float)
    row_weight = degree.copy()
    if multiplicity is not None:
        row_weight *= np.asarray(multiplicity, dtype=float)
    total = max(float(np.sum(row_weight)), EPS)
    field_mean = float(np.sum(row_weight * field) / total)
    centered = field - field_mean
    coefficients = phi.T @ (row_weight * centered)
    # Under resampling, normalize every fixed orbital in the resampled
    # L2(D) inner product.  In the original graph this denominator is one.
    orbital_norm = np.sqrt(np.maximum(
        np.sum(row_weight[:, None] * phi * phi, axis=0), EPS
    ))
    coefficients /= orbital_norm
    power = coefficients * coefficients
    raw = power

    # Within an exactly degenerate eigenspace individual eigenvectors can
    # rotate.  Replace individual raw weights by the block-average so every
    # cutoff and effective-rank calculation is basis invariant in that block.
    invariant = raw.copy()
    for block in np.unique(block_id):
        members = np.flatnonzero(block_id == block)
        invariant[members] = float(np.sum(raw[members])) / len(members)
    total_raw = float(np.sum(invariant))
    if not np.isfinite(total_raw) or total_raw <= EPS:
        raise RuntimeError("Predictive field has no resolved orbital projection.")
    return coefficients, power, invariant, invariant / total_raw, field_mean


def select_predictive_field_dimension(
    phi, eigenvalues, degree, posterior, priors, checkpoints=(16, 32, 64),
    k_min=2, bootstrap_repetitions=200, random_state=42,
):
    phi = np.asarray(phi, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    degree = np.asarray(degree, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != len(eigenvalues):
        raise ValueError("Orbital matrix/eigenvalue shape mismatch.")
    if len(phi) != len(degree) or len(phi) != len(posterior):
        raise ValueError("Graph field arrays have incompatible lengths.")
    if len(eigenvalues) < int(k_min):
        raise ValueError("Not enough orbitals for automatic dimension selection.")
    field = prior_corrected_log_odds_field(posterior, priors)
    block_id = _spectral_blocks(eigenvalues)
    coefficients, power, weights, q, field_mean = _weights_from_row_multiplicity(
        phi, field, degree, block_id
    )
    selected, participation, mass_k, cumulative = _select_prefix(
        q, block_id, k_min=k_min
    )
    entropy = float(np.exp(-np.sum(q * np.log(np.maximum(q, EPS)))))

    usable_checkpoints = sorted(set(
        min(len(eigenvalues), max(int(k_min), int(value)))
        for value in checkpoints if int(value) >= int(k_min)
    ))
    if not usable_checkpoints or usable_checkpoints[-1] != len(eigenvalues):
        usable_checkpoints.append(len(eigenvalues))
    checkpoint_records = []
    for width in usable_checkpoints:
        partial_blocks = block_id[:width]
        _, _, _, q_partial, _ = _weights_from_row_multiplicity(
            phi[:, :width], field, degree, partial_blocks
        )
        partial_k, partial_rank, _, partial_cumulative = _select_prefix(
            q_partial, partial_blocks, k_min=k_min
        )
        checkpoint_records.append({
            "computed_orbitals": int(width),
            "automatic_K_PF": int(partial_k),
            "participation_rank": float(partial_rank),
            "captured_field_power": float(partial_cumulative[partial_k - 1]),
            "touches_checkpoint_ceiling": bool(partial_k >= width),
        })
    recent = checkpoint_records[-2:]
    spectrum_stable = bool(
        len(recent) == 2
        and recent[0]["automatic_K_PF"] == recent[1]["automatic_K_PF"]
        and not recent[1]["touches_checkpoint_ceiling"]
    )

    rng = np.random.RandomState(int(random_state))
    bootstrap_k = []
    for _ in range(max(0, int(bootstrap_repetitions))):
        multiplicity = rng.multinomial(len(phi), np.full(len(phi), 1.0 / len(phi)))
        try:
            _, _, _, q_boot, _ = _weights_from_row_multiplicity(
                phi, field, degree, block_id, multiplicity=multiplicity
            )
            k_boot, _, _, _ = _select_prefix(q_boot, block_id, k_min=k_min)
            bootstrap_k.append(k_boot)
        except RuntimeError:
            continue
    return PredictiveFieldDimensionResult(
        selected_k=int(selected),
        participation_rank=participation,
        entropy_rank=entropy,
        mass_location_k=int(mass_k),
        cutoff_block_end=int(selected),
        captured_weight=float(cumulative[selected - 1]),
        field_weighted_mean=float(field_mean),
        coefficients=coefficients,
        coefficient_power=power,
        predictive_weights=weights,
        normalized_weights=q,
        cumulative_weights=cumulative,
        block_id=block_id,
        bootstrap_k=np.asarray(bootstrap_k, dtype=int),
        checkpoint_records=tuple(checkpoint_records),
        spectrum_stable=spectrum_stable,
    )


def predictive_risk_audit(X_train_reference, X_validation_reference,
                          y_train, y_validation, Phi_train, Phi_validation,
                          selected_k, confidence_level=0.95, probe_C=1.0,
                          probe_max_iter=3000, random_state=42):
    """Audit, but never select, the automatic geometric dimension."""
    y_train = np.asarray(y_train, dtype=int)
    y_validation = np.asarray(y_validation, dtype=int)
    reference_probe = make_probabilistic_probe(
        C=probe_C, max_iter=probe_max_iter, random_state=random_state
    )
    reference_losses, _ = fit_point_losses(
        X_train_reference, y_train, X_validation_reference, y_validation,
        reference_probe,
    )
    rows = []
    z_value = float(norm.ppf(confidence_level))
    maximum = min(Phi_train.shape[1], Phi_validation.shape[1])
    for K in range(2, maximum + 1):
        probe = make_probabilistic_probe(
            C=probe_C, max_iter=probe_max_iter,
            random_state=random_state,
        )
        losses, probability = fit_point_losses(
            normalized_orbital_state(Phi_train, K),
            y_train,
            normalized_orbital_state(Phi_validation, K),
            y_validation,
            probe,
        )
        difference = losses - reference_losses
        standard_error = (
            float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1 else 0.0
        )
        mean_difference = float(np.mean(difference))
        rows.append({
            "K": int(K),
            "selected_by_predictive_geometry": bool(K == int(selected_k)),
            "R_K": float(np.mean(losses)),
            "R_X": float(np.mean(reference_losses)),
            "delta_I_hat": mean_difference,
            "delta_I_standard_error": standard_error,
            "delta_I_upper_confidence": float(
                mean_difference + z_value * standard_error
            ),
            "validation_auc": float(
                roc_auc_score(y_validation, probability[:, 1])
            ),
        })
    selected = next(row for row in rows if row["K"] == int(selected_k))
    return rows, selected


def sparse_selected_risk_audit(
    X_train_reference,
    X_validation_reference,
    y_train,
    y_validation,
    Phi_train,
    Phi_validation,
    selected_indices,
    confidence_level=0.95,
    probe_C=1.0,
    probe_max_iter=3000,
    random_state=42,
):
    """Post-selection audit for a non-contiguous orbital set.

    The logistic probe and labels in this audit never choose the orbital set.
    """
    y_train = np.asarray(y_train, dtype=int)
    y_validation = np.asarray(y_validation, dtype=int)
    indices = np.asarray(selected_indices, dtype=int)
    reference_probe = make_probabilistic_probe(
        C=probe_C, max_iter=probe_max_iter, random_state=random_state
    )
    reference_losses, _ = fit_point_losses(
        X_train_reference,
        y_train,
        X_validation_reference,
        y_validation,
        reference_probe,
    )
    probe = make_probabilistic_probe(
        C=probe_C, max_iter=probe_max_iter, random_state=random_state
    )
    losses, probability = fit_point_losses(
        normalized_selected_orbital_state(Phi_train, indices),
        y_train,
        normalized_selected_orbital_state(Phi_validation, indices),
        y_validation,
        probe,
    )
    difference = losses - reference_losses
    standard_error = (
        float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
        if len(difference) > 1 else 0.0
    )
    mean_difference = float(np.mean(difference))
    row = {
        "K": int(len(indices)),
        "selected_orbitals_one_based": ";".join(
            str(int(value + 1)) for value in indices
        ),
        "selected_by_derivative_information": True,
        "audit_selects_dimension": False,
        "R_K": float(np.mean(losses)),
        "R_X": float(np.mean(reference_losses)),
        "delta_I_hat": mean_difference,
        "delta_I_standard_error": standard_error,
        "delta_I_upper_confidence": float(
            mean_difference + float(norm.ppf(confidence_level)) * standard_error
        ),
        "validation_auc": float(roc_auc_score(y_validation, probability[:, 1])),
    }
    return [row], row
