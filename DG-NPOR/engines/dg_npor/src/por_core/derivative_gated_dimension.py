"""Automatic sparse orbital selection from derivatives of PDF information.

The POR operator and its eigenfunctions are fixed before this module acts.
For a normalized orbital state ``a`` and a common-completeness PSD screening
measurement, an orbital gate ``z_k`` is inserted before re-normalization:

    b(z) = (z * a) / ||z * a||,
    p(z) = b(z)^T M_1 b(z),
    J(z) = KL((1-p(z), p(z)) || (pi_0, pi_1)).

Importance is the mean squared analytic derivative dJ/dz_k at z=1.  It is
therefore not a derivative of classification loss.  Exactly/near-degenerate
eigenspaces receive a common gate so selection is invariant to basis rotation
inside the block.
"""

from dataclasses import dataclass

import numpy as np

from .orbitals import normalized_orbital_state


EPS = 1e-12


def _sym_inverse_sqrt(matrix, floor=1e-9):
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, float(floor))
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def spectral_blocks(eigenvalues):
    """Numerically rotation-safe spectral blocks."""
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


def class_occupancy_screening_measurement(A, y, rho=1e-3):
    """Construct a loss-free PSD measurement from class orbital occupancy.

    Each class scatter is normalized by its own event count.  Hence this
    screening object estimates class-conditional density rather than learning
    a classifier through NLL.  Congruence normalization guarantees M_c >= 0
    and sum_c M_c = I.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if A.ndim != 2 or len(A) != len(y) or len(classes) != 2:
        raise ValueError("Binary aligned orbital states are required.")
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("Screening measurement expects encoded labels 0 and 1.")
    K = A.shape[1]
    identity = np.eye(K)
    E = []
    for class_index in classes:
        state = A[y == class_index]
        if not len(state):
            raise ValueError("Every class must be represented.")
        scatter = state.T @ state / float(len(state))
        E.append(scatter + (float(rho) / len(classes)) * identity)
    E = np.asarray(E)
    inverse_root = _sym_inverse_sqrt(np.sum(E, axis=0))
    M = np.asarray([
        inverse_root @ operator @ inverse_root for operator in E
    ])
    M = 0.5 * (M + np.swapaxes(M, 1, 2))
    return M


def predictive_information_gate_gradient(A, signal_operator, priors):
    """Analytic dJ/dz_k for independently gated normalized orbitals.

    At z=1,

      dp/dz_k = 2 a_k [(M_1 a)_k - p a_k]
      dJ/dp   = log[p pi_0 / ((1-p) pi_1)].
    """
    A = np.asarray(A, dtype=float)
    M = np.asarray(signal_operator, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if A.ndim != 2 or M.shape != (A.shape[1], A.shape[1]):
        raise ValueError("State/operator dimensions do not match.")
    if priors.shape != (2,) or np.any(priors <= 0):
        raise ValueError("Positive binary class priors are required.")
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
    probability = np.einsum("ni,ij,nj->n", A, M, A)
    probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
    operator_state = A @ M.T
    dp_dz = 2.0 * A * (
        operator_state - probability[:, None] * A
    )
    dJ_dp = np.log(
        (probability * priors[0])
        / ((1.0 - probability) * priors[1])
    )
    return dJ_dp[:, None] * dp_dz, probability


def _block_gradients(individual_gradient, block_id):
    blocks = np.unique(block_id)
    output = np.empty((len(individual_gradient), len(blocks)), dtype=float)
    members = []
    for column, block in enumerate(blocks):
        indices = np.flatnonzero(block_id == block)
        members.append(indices)
        # A common block gate scales every basis vector in the eigenspace.
        output[:, column] = np.sum(individual_gradient[:, indices], axis=1)
    return output, members


def _selection_from_scores(block_scores, block_members, k_min=2):
    scores = np.asarray(block_scores, dtype=float)
    scores = np.maximum(scores, 0.0)
    total = float(np.sum(scores))
    if not np.isfinite(total) or total <= EPS:
        raise RuntimeError("Predictive-information derivative is unresolved.")
    weights = scores / total
    participation = float(1.0 / max(np.sum(weights * weights), EPS))
    entropy = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, EPS)))))
    target = max(0.0, 1.0 - 1.0 / max(participation, 1.0))
    order = np.asarray(sorted(
        range(len(weights)),
        key=lambda index: (-weights[index], int(block_members[index][0])),
    ), dtype=int)
    required_blocks = max(1, int(np.ceil(participation - 1e-12)))
    selected_blocks = []
    captured = 0.0
    dimensions = 0
    for block in order:
        selected_blocks.append(int(block))
        captured += float(weights[block])
        dimensions += int(len(block_members[block]))
        if (
            len(selected_blocks) >= required_blocks
            and dimensions >= int(k_min)
            and captured >= target
        ):
            break
    selected_indices = np.sort(np.concatenate([
        block_members[block] for block in selected_blocks
    ])).astype(int)
    return {
        "weights": weights,
        "participation": participation,
        "entropy": entropy,
        "target": float(target),
        "order": order,
        "selected_blocks": np.asarray(selected_blocks, dtype=int),
        "selected_indices": selected_indices,
        "captured": float(captured),
    }


def _single_width(phi_train, phi_validation, eigenvalues, y_train, priors,
                  rho, k_min):
    width = len(eigenvalues)
    train_state = normalized_orbital_state(phi_train, width)
    validation_state = normalized_orbital_state(phi_validation, width)
    measurement = class_occupancy_screening_measurement(
        train_state, y_train, rho=rho
    )
    individual_gradient, probability = predictive_information_gate_gradient(
        validation_state, measurement[1], priors
    )
    block_id = spectral_blocks(eigenvalues)
    block_gradient, block_members = _block_gradients(
        individual_gradient, block_id
    )
    block_scores = np.mean(block_gradient * block_gradient, axis=0)
    selection = _selection_from_scores(
        block_scores, block_members, k_min=k_min
    )
    return {
        "measurement": measurement,
        "individual_gradient": individual_gradient,
        "probability": probability,
        "block_id": block_id,
        "block_gradient": block_gradient,
        "block_members": block_members,
        "block_scores": block_scores,
        **selection,
    }


@dataclass(frozen=True)
class DerivativeGatedDimensionResult:
    selected_indices: np.ndarray
    ranked_indices: np.ndarray
    orbital_scores: np.ndarray
    orbital_weights: np.ndarray
    orbital_rank: np.ndarray
    block_id: np.ndarray
    participation_rank: float
    entropy_rank: float
    target_mass: float
    captured_weight: float
    screening_probability_mean: float
    screening_rho: float
    screening_completeness_max_abs_error: float
    screening_minimum_eigenvalue: float
    bootstrap_k: np.ndarray
    bootstrap_exact_set_fraction: float
    bootstrap_inclusion_frequency: np.ndarray
    checkpoint_records: tuple
    spectrum_stable: bool
    stability_source: str

    @property
    def selected_k(self):
        return int(len(self.selected_indices))

    def summary(self):
        if len(self.bootstrap_k):
            values, counts = np.unique(self.bootstrap_k, return_counts=True)
            mode = int(values[np.argmax(counts)])
            mode_fraction = float(np.max(counts) / len(self.bootstrap_k))
            exact_k = float(np.mean(self.bootstrap_k == self.selected_k))
        else:
            mode = None
            mode_fraction = None
            exact_k = None
        return {
            "selection_method": (
                "derivative_gated_PDF_predictive_information_sparse_orbitals"
            ),
            "derivative_target": "J=KL(P(Y|x)||P(Y))",
            "classification_loss_derivative_used": False,
            "screening_measurement": (
                "class_conditional_orbital_occupancy_PSD_no_loss_optimization"
            ),
            "user_selected_K": False,
            "user_selected_delta": False,
            "K_DG": self.selected_k,
            "selected_orbitals_one_based": [
                int(value + 1) for value in self.selected_indices
            ],
            "max_selected_orbital_index": int(np.max(self.selected_indices) + 1),
            "noncontiguous_selection": bool(
                not np.array_equal(
                    self.selected_indices,
                    np.arange(self.selected_k, dtype=int),
                )
            ),
            "derivative_participation_rank": float(self.participation_rank),
            "derivative_entropy_rank_diagnostic": float(self.entropy_rank),
            "data_derived_target_mass": float(self.target_mass),
            "captured_derivative_information": float(self.captured_weight),
            "screening_signal_probability_mean": float(
                self.screening_probability_mean
            ),
            "screening_rho": float(self.screening_rho),
            "screening_completeness_max_abs_error": float(
                self.screening_completeness_max_abs_error
            ),
            "screening_minimum_eigenvalue": float(
                self.screening_minimum_eigenvalue
            ),
            "degeneracy_safe_block_gating": True,
            "spectrum_checkpoint_stable": bool(self.spectrum_stable),
            "spectrum_stability_source": str(self.stability_source),
            "bootstrap_repetitions": int(len(self.bootstrap_k)),
            "bootstrap_scope": (
                "dimension_gate_rows_conditional_on_frozen_graph_and_screening_PSD"
            ),
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
            "bootstrap_K_mode": mode,
            "bootstrap_K_mode_fraction": mode_fraction,
            "bootstrap_exact_selected_K_fraction": exact_k,
            "bootstrap_exact_selected_orbital_set_fraction": float(
                self.bootstrap_exact_set_fraction
            ),
        }

    def records(self, eigenvalues):
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        return [
            {
                "orbital": int(index + 1),
                "eigenvalue": float(eigenvalues[index]),
                "block_id": int(self.block_id[index]),
                "dJ_dgate_squared_mean": float(self.orbital_scores[index]),
                "normalized_derivative_weight": float(
                    self.orbital_weights[index]
                ),
                "derivative_importance_rank": int(self.orbital_rank[index]),
                "bootstrap_inclusion_frequency": float(
                    self.bootstrap_inclusion_frequency[index]
                ),
                "selected": bool(index in set(self.selected_indices.tolist())),
            }
            for index in range(len(eigenvalues))
        ]


def select_derivative_gated_dimension(
    phi_train,
    phi_validation,
    eigenvalues,
    y_train,
    priors,
    checkpoints=(16, 32, 64),
    screening_rho=1e-3,
    k_min=2,
    bootstrap_repetitions=200,
    random_state=42,
    candidate_bank_resolved=False,
):
    """Select a non-contiguous, data-sized orbital set from dJ/dgate."""
    phi_train = np.asarray(phi_train, dtype=float)
    phi_validation = np.asarray(phi_validation, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    priors = np.asarray(priors, dtype=float)
    if phi_train.ndim != 2 or phi_validation.ndim != 2:
        raise ValueError("Train/validation orbitals must be matrices.")
    if phi_train.shape[1] != len(eigenvalues):
        raise ValueError("Training orbital/eigenvalue dimensions disagree.")
    if phi_validation.shape[1] != len(eigenvalues):
        raise ValueError("Validation orbital/eigenvalue dimensions disagree.")
    if len(phi_train) != len(y_train):
        raise ValueError("Training orbitals and labels are not aligned.")
    if len(eigenvalues) < int(k_min):
        raise ValueError("Not enough orbitals for automatic selection.")

    full = _single_width(
        phi_train,
        phi_validation,
        eigenvalues,
        y_train,
        priors,
        screening_rho,
        k_min,
    )
    block_id = full["block_id"]
    block_members = full["block_members"]
    orbital_scores = np.empty(len(eigenvalues), dtype=float)
    orbital_weights = np.empty(len(eigenvalues), dtype=float)
    for block, members in enumerate(block_members):
        orbital_scores[members] = full["block_scores"][block] / len(members)
        orbital_weights[members] = full["weights"][block] / len(members)
    ranked_indices = np.asarray(sorted(
        range(len(eigenvalues)),
        key=lambda index: (-orbital_weights[index], index),
    ), dtype=int)
    orbital_rank = np.empty(len(eigenvalues), dtype=int)
    orbital_rank[ranked_indices] = np.arange(1, len(eigenvalues) + 1)

    usable = sorted(set(
        min(len(eigenvalues), max(int(k_min), int(value)))
        for value in checkpoints if int(value) >= int(k_min)
    ))
    if not usable or usable[-1] != len(eigenvalues):
        usable.append(len(eigenvalues))
    checkpoint_records = []
    checkpoint_sets = []
    for width in usable:
        partial = _single_width(
            phi_train[:, :width],
            phi_validation[:, :width],
            eigenvalues[:width],
            y_train,
            priors,
            screening_rho,
            k_min,
        )
        selected = partial["selected_indices"]
        checkpoint_sets.append(selected)
        checkpoint_records.append({
            "computed_orbitals": int(width),
            "automatic_K_DG": int(len(selected)),
            "selected_orbitals_one_based": ";".join(
                str(int(value + 1)) for value in selected
            ),
            "max_selected_orbital": int(np.max(selected) + 1),
            "derivative_participation_rank": float(partial["participation"]),
            "captured_derivative_information": float(partial["captured"]),
            "touches_checkpoint_ceiling": bool(np.max(selected) + 1 >= width),
        })
    recent_sets = checkpoint_sets[-2:]
    recent_records = checkpoint_records[-2:]
    derivative_checkpoint_stable = bool(
        len(recent_sets) == 2
        and np.array_equal(recent_sets[0], recent_sets[1])
    )
    # In the estimator the derivative acts inside a separately resolved
    # predictive-field candidate bank.  Selecting its last member is allowed;
    # the outer bank, rather than dJ/dz, controls spectral truncation.
    spectrum_stable = bool(
        candidate_bank_resolved or derivative_checkpoint_stable
    )

    rng = np.random.RandomState(int(random_state))
    bootstrap_k = []
    inclusion = np.zeros(len(eigenvalues), dtype=float)
    exact_set = 0
    squared_block_gradient = full["block_gradient"] ** 2
    selected_full = full["selected_indices"]
    for _ in range(max(0, int(bootstrap_repetitions))):
        multiplicity = rng.multinomial(
            len(phi_validation),
            np.full(len(phi_validation), 1.0 / len(phi_validation)),
        )
        scores = (
            multiplicity @ squared_block_gradient
        ) / max(float(np.sum(multiplicity)), 1.0)
        selection = _selection_from_scores(
            scores, block_members, k_min=k_min
        )
        chosen = selection["selected_indices"]
        bootstrap_k.append(len(chosen))
        inclusion[chosen] += 1.0
        exact_set += int(np.array_equal(chosen, selected_full))
    repetitions = len(bootstrap_k)
    if repetitions:
        inclusion /= float(repetitions)

    return DerivativeGatedDimensionResult(
        selected_indices=np.asarray(selected_full, dtype=int),
        ranked_indices=ranked_indices,
        orbital_scores=orbital_scores,
        orbital_weights=orbital_weights,
        orbital_rank=orbital_rank,
        block_id=block_id,
        participation_rank=float(full["participation"]),
        entropy_rank=float(full["entropy"]),
        target_mass=float(full["target"]),
        captured_weight=float(full["captured"]),
        screening_probability_mean=float(np.mean(full["probability"])),
        screening_rho=float(screening_rho),
        screening_completeness_max_abs_error=float(np.max(np.abs(
            np.sum(full["measurement"], axis=0)
            - np.eye(len(eigenvalues))
        ))),
        screening_minimum_eigenvalue=float(min(
            np.min(np.linalg.eigvalsh(operator))
            for operator in full["measurement"]
        )),
        bootstrap_k=np.asarray(bootstrap_k, dtype=int),
        bootstrap_exact_set_fraction=(
            0.0 if not repetitions else float(exact_set / repetitions)
        ),
        bootstrap_inclusion_frequency=inclusion,
        checkpoint_records=tuple(checkpoint_records),
        spectrum_stable=spectrum_stable,
        stability_source=(
            "resolved_outer_predictive_field_candidate_bank"
            if candidate_bank_resolved
            else (
                "derivative_checkpoint_set_agreement"
                if derivative_checkpoint_stable else "unresolved"
            )
        ),
    )
