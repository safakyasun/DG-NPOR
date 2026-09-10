"""Neural geometry, PDF-core POR, and derivative-gated sparse orbitals.

The spectral operator is exactly H=L_sym+lambda_pi*V_PI.  No directed-flow
term is present.  A training-only h_theta learns the event metric; its
auxiliary probability never enters the final POR prediction.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .risk_audit import make_probabilistic_probe
from .graph import build_self_tuning_graph, query_affinities_sparse
from .information import (
    class_priors,
    leave_one_out_kernel_posterior,
    posterior_from_affinities,
    predictive_information_potential,
)
from .operator import solve_por_eigensystem
from .orbitals import (
    extend_orbitals,
    normalized_orbital_state,
    normalized_selected_orbital_state,
)
from .predictive_field_dimension import (
    sparse_selected_risk_audit,
    select_predictive_field_dimension,
)
from .derivative_gated_dimension import select_derivative_gated_dimension
from .readout import StructuredOrbitalMeasurement
from .neural_geometry import NeuralGeometryMap


@dataclass
class _GraphBase:
    X: np.ndarray
    y: np.ndarray
    priors: np.ndarray
    graph: object
    posterior: np.ndarray
    information: np.ndarray
    potential: np.ndarray


@dataclass
class _OperatorCore:
    base: _GraphBase
    eigensystem: object
    lambda_pi: float


class DerivativeGatedNeuralPOR:
    """Binary Neural-POR with dJ/dgate-selected sparse orbitals."""

    def __init__(
        self,
        n_neighbors=32,
        scale_neighbor=16,
        lambda0=1.0,
        lambda_pi_grid=(0.10, 0.25),
        k_max=64,
        dimension_bootstrap_repetitions=200,
        screening_rho=1e-3,
        confidence_level=0.95,
        probe_C=1.0,
        probe_max_iter=3000,
        denominator_floor=1e-5,
        measurement_rank=4,
        measurement_rho=1e-3,
        measurement_l2=1e-4,
        measurement_max_iter=1200,
        measurement_restarts=2,
        max_measurement_samples=8000,
        max_graph_samples=20000,
        query_batch_size=2048,
        neural_max_epochs=250,
        neural_patience=25,
        neural_learning_rate=1e-3,
        neural_alpha=1e-4,
        neural_batch_size=256,
        neural_architecture="standard",
        random_state=42,
    ):
        self.n_neighbors = int(n_neighbors)
        self.scale_neighbor = int(scale_neighbor)
        self.lambda0 = float(lambda0)
        self.lambda_pi_grid = tuple(float(value) for value in lambda_pi_grid)
        self.k_max = int(k_max)
        self.dimension_bootstrap_repetitions = int(
            dimension_bootstrap_repetitions
        )
        self.screening_rho = float(screening_rho)
        self.confidence_level = float(confidence_level)
        self.probe_C = float(probe_C)
        self.probe_max_iter = int(probe_max_iter)
        self.denominator_floor = float(denominator_floor)
        self.measurement_rank = int(measurement_rank)
        self.measurement_rho = float(measurement_rho)
        self.measurement_l2 = float(measurement_l2)
        self.measurement_max_iter = int(measurement_max_iter)
        self.measurement_restarts = int(measurement_restarts)
        self.max_measurement_samples = (
            None if max_measurement_samples is None else int(max_measurement_samples)
        )
        self.max_graph_samples = (
            None if max_graph_samples is None else int(max_graph_samples)
        )
        self.query_batch_size = int(query_batch_size)
        self.neural_max_epochs = int(neural_max_epochs)
        self.neural_patience = int(neural_patience)
        self.neural_learning_rate = float(neural_learning_rate)
        self.neural_alpha = float(neural_alpha)
        self.neural_batch_size = int(neural_batch_size)
        self.neural_architecture = str(neural_architecture)
        self.random_state = int(random_state)
        self._validate_parameters()

    def _validate_parameters(self):
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least two.")
        if self.scale_neighbor < 1 or self.scale_neighbor > self.n_neighbors:
            raise ValueError("scale_neighbor must lie in [1, n_neighbors].")
        if self.lambda0 <= 0:
            raise ValueError("lambda0 must be positive.")
        if not self.lambda_pi_grid or min(self.lambda_pi_grid) < 0:
            raise ValueError("lambda_pi_grid must be non-empty and non-negative.")
        if self.k_max < 2:
            raise ValueError("k_max must be at least two.")
        if self.dimension_bootstrap_repetitions < 0:
            raise ValueError("dimension_bootstrap_repetitions cannot be negative.")
        if self.screening_rho <= 0:
            raise ValueError("screening_rho must be positive.")

    def _prepare_base(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        graph = build_self_tuning_graph(
            X,
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            scale_neighbor=min(self.scale_neighbor, len(X) - 1),
        )
        priors = class_priors(y)
        posterior = leave_one_out_kernel_posterior(
            graph.W, y, priors, lambda0=self.lambda0
        )
        predictive = predictive_information_potential(posterior, priors)
        return _GraphBase(
            X, y, priors, graph, posterior, predictive.J, predictive.V
        )

    @staticmethod
    def _dimension_checkpoints(width):
        width = int(width)
        if width < 4:
            return (2, width)
        values = [min(16, max(2, width // 2))]
        while values[-1] < width:
            values.append(min(width, 2 * values[-1]))
        if len(values) == 1:
            values.insert(0, max(2, width // 2))
        return tuple(sorted(set(values)))

    def _fit_operator(self, base, parameters, n_components):
        lambda_pi = float(parameters)
        eigensystem = solve_por_eigensystem(
            base.graph.W,
            base.graph.degree,
            base.potential,
            n_components=min(int(n_components), len(base.X) - 1),
            alpha=1.0,
            beta=lambda_pi,
            random_state=self.random_state,
        )
        return _OperatorCore(base, eigensystem, lambda_pi)

    def _extend(self, core, X_query, K):
        affinities, degree = query_affinities_sparse(
            core.base.graph, np.asarray(X_query, dtype=float)
        )
        posterior = posterior_from_affinities(
            affinities, core.base.y, core.base.priors, lambda0=self.lambda0
        )
        potential = predictive_information_potential(
            posterior, core.base.priors
        ).V
        Phi, stable = extend_orbitals(
            affinities,
            degree,
            core.eigensystem.phi[:, :K],
            core.eigensystem.eigenvalues[:K],
            potential,
            alpha=1.0,
            beta=core.lambda_pi,
            denominator_floor=self.denominator_floor,
        )
        return np.asarray(Phi[:, :K], dtype=float), np.asarray(stable[:K], dtype=bool)

    @staticmethod
    def _stable_prefix(stable):
        prefix = 0
        for value in np.asarray(stable, dtype=bool):
            if not value:
                break
            prefix += 1
        return prefix

    def _subsample_aligned(self, X, y, reference, seed):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        reference = np.asarray(reference, dtype=float)
        indices = np.arange(len(X), dtype=int)
        limited = bool(
            self.max_graph_samples is not None
            and self.max_graph_samples > 0
            and len(X) > self.max_graph_samples
        )
        if limited:
            X, _, y, _, reference, _, indices, _ = train_test_split(
                X,
                y,
                reference,
                indices,
                train_size=self.max_graph_samples,
                stratify=y,
                random_state=int(seed),
            )
        return X, y, reference, indices, limited

    def _operator_record(self, parameters, K, stable_prefix, y_validation,
                         probability, dimension):
        prediction = np.argmax(probability, axis=1)
        return {
            "lambda_pi": float(parameters),
            "probe_K": int(K),
            "automatic_K_PF": int(K),
            "participation_rank": float(dimension.participation_rank),
            "captured_predictive_field_power": float(
                dimension.captured_weight
            ),
            "spectrum_checkpoint_stable": bool(dimension.spectrum_stable),
            "stable_prefix": int(stable_prefix),
            "validation_log_loss": float(
                log_loss(y_validation, probability, labels=[0, 1])
            ),
            "validation_auc": float(roc_auc_score(y_validation, probability[:, 1])),
            "validation_balanced_accuracy_at_0_5": float(
                balanced_accuracy_score(y_validation, prediction)
            ),
        }

    def fit(
        self,
        X,
        y,
        X_operator_validation,
        y_operator_validation,
        X_dimension_validation,
        y_dimension_validation,
        X_reference_train=None,
        X_reference_dimension=None,
    ):
        """Fit with disjoint graph, operator, derivative, and test roles.

        Orbital gates are selected from dJ/dz on the dimension split.  Its
        labels are not used by the derivative calculation.  The same split
        may enter the final PSD training only after the orbital set is frozen.
        Test observations must never enter.
        """
        X_physics = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X_operator_physics = np.asarray(X_operator_validation, dtype=float)
        y_operator_validation = np.asarray(y_operator_validation)
        X_dimension_physics = np.asarray(X_dimension_validation, dtype=float)
        y_dimension_validation = np.asarray(y_dimension_validation)
        if X_reference_train is None:
            X_reference_train = X_physics
        if X_reference_dimension is None:
            X_reference_dimension = X_dimension_physics
        X_reference_train = np.asarray(X_reference_train, dtype=float)
        X_reference_dimension = np.asarray(X_reference_dimension, dtype=float)

        if len(X_physics) != len(y) or len(X_physics) != len(X_reference_train):
            raise ValueError("Training arrays have incompatible lengths.")
        if len(X_operator_physics) != len(y_operator_validation):
            raise ValueError("Operator-validation arrays have incompatible lengths.")
        if len(X_dimension_physics) != len(y_dimension_validation):
            raise ValueError("Dimension-audit arrays have incompatible lengths.")
        if len(X_reference_dimension) != len(y_dimension_validation):
            raise ValueError("Dimension-audit reference length mismatch.")

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) != 2:
            raise ValueError("Compact POR currently supports binary classification.")
        y_encoded = self.label_encoder_.transform(y)
        y_operator_encoded = self.label_encoder_.transform(y_operator_validation)
        y_dimension_encoded = self.label_encoder_.transform(y_dimension_validation)

        self.neural_geometry_ = NeuralGeometryMap(
            max_epochs=self.neural_max_epochs,
            patience=self.neural_patience,
            learning_rate=self.neural_learning_rate,
            alpha=self.neural_alpha,
            batch_size=self.neural_batch_size,
            architecture=self.neural_architecture,
            random_state=self.random_state,
        ).fit(X_physics, y_encoded)
        X = self.neural_geometry_.transform(X_physics)
        X_operator_validation = self.neural_geometry_.transform(
            X_operator_physics
        )
        X_dimension_validation = self.neural_geometry_.transform(
            X_dimension_physics
        )

        X_graph, y_graph, reference_graph, indices, limited = self._subsample_aligned(
            X,
            y_encoded,
            X_reference_train,
            self.random_state,
        )
        self.selection_training_indices_ = np.asarray(indices, dtype=int)
        self.selection_graph_subsampled_ = limited
        self.n_selection_graph_samples_ = int(len(X_graph))
        base = self._prepare_base(X_graph, y_graph)
        available_k = min(self.k_max, len(X_graph) - 1)

        candidates = []
        records = []
        for parameters in self.lambda_pi_grid:
            core = self._fit_operator(base, parameters, available_k)
            Phi_validation, stable = self._extend(
                core, X_operator_validation, available_k
            )
            stable_prefix = self._stable_prefix(stable)
            if stable_prefix < 2:
                continue
            candidate_dimension = select_predictive_field_dimension(
                phi=core.eigensystem.phi[:, :stable_prefix],
                eigenvalues=core.eigensystem.eigenvalues[:stable_prefix],
                degree=core.base.graph.degree,
                posterior=core.base.posterior,
                priors=core.base.priors,
                checkpoints=self._dimension_checkpoints(stable_prefix),
                k_min=2,
                bootstrap_repetitions=0,
                random_state=self.random_state,
            )
            probe_k = int(candidate_dimension.selected_k)
            if probe_k >= stable_prefix:
                # The automatic support touches the numerical spectrum
                # boundary, so this lambda candidate has unresolved dimension.
                records.append({
                    "lambda_pi": float(parameters),
                    "automatic_K_PF": probe_k,
                    "stable_prefix": int(stable_prefix),
                    "spectral_ceiling_resolved": False,
                    "spectrum_checkpoint_stable": bool(
                        candidate_dimension.spectrum_stable
                    ),
                    "validation_log_loss": None,
                    "validation_auc": None,
                    "validation_balanced_accuracy_at_0_5": None,
                })
                continue
            A_train = normalized_orbital_state(core.eigensystem.phi, probe_k)
            A_validation = normalized_orbital_state(Phi_validation, probe_k)
            probe = make_probabilistic_probe(
                C=self.probe_C,
                max_iter=self.probe_max_iter,
                random_state=self.random_state,
            )
            probe.fit(A_train, y_graph)
            probability = probe.predict_proba(A_validation)
            record = self._operator_record(
                parameters,
                probe_k,
                stable_prefix,
                y_operator_encoded,
                probability,
                candidate_dimension,
            )
            record["spectral_ceiling_resolved"] = True
            records.append(record)
            candidates.append((record, parameters, core))
        if not candidates:
            raise RuntimeError("No operator has a stable two-orbital prefix.")

        # Lambda is an operator hyperparameter selected before automatic
        # dimension inference.  The embedding size itself is not performance-tuned.
        selected_record, selected_parameters, selected_core = min(
            candidates,
            key=lambda item: (
                item[0]["validation_log_loss"],
                -item[0]["validation_auc"],
                item[0]["probe_K"],
            ),
        )
        self.operator_selection_records_ = records
        self.selected_operator_ = dict(selected_record)
        self.selection_eigenvalues_ = np.asarray(
            selected_core.eigensystem.eigenvalues, dtype=float
        )

        Phi_dimension, stable_dimension = self._extend(
            selected_core, X_dimension_validation, available_k
        )
        stable_prefix = min(
            self._stable_prefix(stable_dimension),
            selected_core.eigensystem.phi.shape[1],
        )
        if stable_prefix < 2:
            raise RuntimeError("No stable non-degenerate prefix on the dimension audit split.")

        # Stage 1: resolve a finite low-frequency candidate bank from the
        # unpenalized PDF predictive field.  This prevents derivative screening
        # from chasing modes at an arbitrary k_max boundary.
        self.candidate_bank_result_ = select_predictive_field_dimension(
            phi=selected_core.eigensystem.phi[:, :stable_prefix],
            eigenvalues=selected_core.eigensystem.eigenvalues[:stable_prefix],
            degree=selected_core.base.graph.degree,
            posterior=selected_core.base.posterior,
            priors=selected_core.base.priors,
            checkpoints=self._dimension_checkpoints(stable_prefix),
            k_min=2,
            bootstrap_repetitions=self.dimension_bootstrap_repetitions,
            random_state=self.random_state + 211,
        )
        self.candidate_bank_k_ = int(self.candidate_bank_result_.selected_k)
        self.candidate_bank_resolved_ = bool(
            self.candidate_bank_result_.spectrum_stable
            and self.candidate_bank_k_ < stable_prefix
        )

        # Stage 2: inside the resolved bank only, select a sparse, potentially
        # non-contiguous set from the analytic derivative of PDF J.
        self.dimension_result_ = select_derivative_gated_dimension(
            phi_train=selected_core.eigensystem.phi[:, :self.candidate_bank_k_],
            phi_validation=Phi_dimension[:, :self.candidate_bank_k_],
            eigenvalues=selected_core.eigensystem.eigenvalues[
                :self.candidate_bank_k_
            ],
            y_train=y_graph,
            priors=selected_core.base.priors,
            checkpoints=(self.candidate_bank_k_,),
            screening_rho=self.screening_rho,
            k_min=2,
            bootstrap_repetitions=self.dimension_bootstrap_repetitions,
            random_state=self.random_state + 307,
            candidate_bank_resolved=self.candidate_bank_resolved_,
        )
        self.selected_orbital_indices_ = np.asarray(
            self.dimension_result_.selected_indices, dtype=int
        )
        self.selected_k_ = int(self.dimension_result_.selected_k)
        self.k_dg_ = self.selected_k_
        self.required_orbital_bank_ = int(
            np.max(self.selected_orbital_indices_) + 1
        )
        self.criterion_reached_ = bool(
            self.candidate_bank_resolved_
            and self.required_orbital_bank_ <= self.candidate_bank_k_
        )
        self.dimension_stable_prefix_ = int(stable_prefix)
        self.selected_operator_parameters_ = float(selected_parameters)
        self.reference_dimension_ = int(X_reference_train.shape[1])
        self.strict_reference_compression_ = bool(
            self.selected_k_ < self.reference_dimension_
        )
        self.risk_audit_path_, self.selected_risk_audit_ = sparse_selected_risk_audit(
            X_train_reference=reference_graph,
            X_validation_reference=X_reference_dimension,
            y_train=y_graph,
            y_validation=y_dimension_encoded,
            Phi_train=selected_core.eigensystem.phi[:, :stable_prefix],
            Phi_validation=Phi_dimension[:, :stable_prefix],
            selected_indices=self.selected_orbital_indices_,
            confidence_level=self.confidence_level,
            probe_C=self.probe_C,
            probe_max_iter=self.probe_max_iter,
            random_state=self.random_state,
        )

        # Sparse orbital identities are tied to one fitted operator.  Unlike a
        # prefix dimension, the set {phi_1, phi_18, ...} must not be transferred
        # by index to an independently refitted graph.  Freeze selected_core.
        self.core_ = selected_core
        self.training_indices_ = self.selection_training_indices_
        self.landmark_indices_ = self.training_indices_
        self.graph_subsampled_ = self.selection_graph_subsampled_
        self.n_graph_samples_ = self.n_selection_graph_samples_
        self.n_landmark_samples_ = self.n_graph_samples_

        # Use only observations outside the landmark graph for final PSD
        # training: unused base rows plus operator and dimension partitions.
        base_mask = np.ones(len(X), dtype=bool)
        base_mask[self.selection_training_indices_] = False
        X_measurement_pool = np.vstack([
            X[base_mask], X_operator_validation, X_dimension_validation
        ])
        y_measurement_pool = np.concatenate([
            y_encoded[base_mask], y_operator_encoded, y_dimension_encoded
        ])
        measurement_indices = np.arange(len(X_measurement_pool), dtype=int)
        if (
            self.max_measurement_samples is not None
            and self.max_measurement_samples > 0
            and len(measurement_indices) > self.max_measurement_samples
        ):
            measurement_indices, _ = train_test_split(
                measurement_indices,
                train_size=self.max_measurement_samples,
                stratify=y_measurement_pool[measurement_indices],
                random_state=self.random_state + 2027,
            )
        self.measurement_training_indices_ = np.asarray(
            measurement_indices, dtype=int
        )
        self.measurement_disjoint_from_landmarks_ = True
        raw_measurement = self._extend_in_chunks(
            self.core_,
            X_measurement_pool[self.measurement_training_indices_],
            self.required_orbital_bank_,
        )
        self.A_train_ = normalized_selected_orbital_state(
            raw_measurement, self.selected_orbital_indices_
        )
        self.y_train_encoded_ = np.asarray(
            y_measurement_pool[self.measurement_training_indices_], dtype=int
        )
        self.n_measurement_samples_ = int(len(self.A_train_))
        self.measurement_ = StructuredOrbitalMeasurement(
            rank=self.measurement_rank,
            rho=self.measurement_rho,
            l2=self.measurement_l2,
            max_iter=self.measurement_max_iter,
            n_restarts=self.measurement_restarts,
            random_state=self.random_state,
        ).fit(
            self.A_train_,
            self.y_train_encoded_,
        )
        self.selected_ = self._selection_summary()
        return self

    def _selection_summary(self):
        risk = self.selected_risk_audit_
        dimension = self.dimension_result_.summary()
        return {
            "method_variant": "Derivative-Gated-Neural-POR-J-extension",
            "pdf_core_preserved": True,
            "neural_extension": "training-only h_theta geometry map",
            "dimension_extension": (
                "analytic derivative of PDF J with respect to orbital gates"
            ),
            "neural_geometry": self.neural_geometry_.summary(),
            "neural_auxiliary_probability_used_for_final_prediction": False,
            "operator": "H = L_sym + lambda_pi*V_PI",
            "lambda_pi": float(self.selected_operator_parameters_),
            "gamma_L_dir_present": False,
            "classification_loss_derivative_used": False,
            "dimension_selected_by_validation_performance": False,
            "user_selected_K": False,
            "user_selected_delta": False,
            "K_DG": int(self.selected_k_),
            "selected_K": int(self.selected_k_),
            "candidate_bank_method": "predictive_field_spectral_support",
            "candidate_K_PF": int(self.candidate_bank_k_),
            "candidate_bank_resolved": bool(self.candidate_bank_resolved_),
            "candidate_spectrum_checkpoint_stable": bool(
                self.candidate_bank_result_.spectrum_stable
            ),
            "candidate_bank_diagnostics": self.candidate_bank_result_.summary(),
            "selected_orbitals_one_based": [
                int(value + 1) for value in self.selected_orbital_indices_
            ],
            "required_orbital_bank": int(self.required_orbital_bank_),
            "derivative_selection_restricted_to_candidate_bank": True,
            "stable_prefix": int(self.dimension_stable_prefix_),
            "reference_dimension": int(self.reference_dimension_),
            "strict_reference_compression": bool(self.strict_reference_compression_),
            "compression_ratio_to_reference": float(
                self.selected_k_ / self.reference_dimension_
            ),
            **dimension,
            "risk_audit_selects_K": False,
            "dimension_resolved": bool(self.criterion_reached_),
            "R_X_audit": float(risk["R_X"]),
            "R_K_DG_audit": float(risk["R_K"]),
            "delta_I_hat_audit": float(risk["delta_I_hat"]),
            "delta_I_upper_confidence_audit": float(
                risk["delta_I_upper_confidence"]
            ),
            "validation_auc_audit": float(risk["validation_auc"]),
            "final_state": "normalized_selected_orbitals_only",
            "final_readout": "structured_PSD_with_cross_orbital_terms",
            "probability_calibration": "none",
            "landmark_graph_rows": getattr(self, "n_landmark_samples_", None),
            "measurement_training_rows": getattr(
                self, "n_measurement_samples_", None
            ),
            "measurement_disjoint_from_landmarks": getattr(
                self, "measurement_disjoint_from_landmarks_", None
            ),
            "measurement_optimizer_success": (
                None if getattr(self, "measurement_", None) is None
                else bool(self.measurement_.optimization_result_.success)
            ),
        }

    def dimension_path(self):
        return self.dimension_result_.records(
            self.selection_eigenvalues_[:self.candidate_bank_k_]
        )

    def risk_audit_path(self):
        return list(self.risk_audit_path_)

    def dimension_checkpoint_path(self):
        return list(self.dimension_result_.checkpoint_records)

    def candidate_bank_checkpoint_path(self):
        return list(self.candidate_bank_result_.checkpoint_records)

    def candidate_bank_path(self):
        return self.candidate_bank_result_.records(
            self.selection_eigenvalues_[:self.dimension_stable_prefix_]
        )

    def neural_training_path(self):
        return list(self.neural_geometry_.history_)

    def _iter_chunks(self, X):
        X = np.asarray(X, dtype=float)
        size = max(1, self.query_batch_size)
        for start in range(0, len(X), size):
            yield X[start:start + size]

    def _extend_in_chunks(self, core, X, K):
        chunks = []
        for block in self._iter_chunks(X):
            Phi, stable = self._extend(core, block, K)
            if not np.all(stable):
                raise RuntimeError(
                    "Selected orbital prefix is unstable on inductive data."
                )
            chunks.append(Phi)
        return np.vstack(chunks) if chunks else np.empty((0, int(K)))

    def transform_raw_orbitals(self, X):
        geometry = self.neural_geometry_.transform(X)
        return self._extend_in_chunks(
            self.core_, geometry, self.required_orbital_bank_
        )

    def transform(self, X):
        return normalized_selected_orbital_state(
            self.transform_raw_orbitals(X), self.selected_orbital_indices_
        )

    def predict_proba(self, X):
        if self.measurement_ is None:
            raise RuntimeError("Structured measurement is unavailable.")
        return self.measurement_.predict_proba(self.transform(X))

    def predict(self, X):
        encoded = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def probability_decomposition(self, X):
        """Separate diagonal occupations from cross-orbital interactions."""
        state = self.transform(X)
        raw_total = np.column_stack([
            np.einsum("ni,ij,nj->n", state, operator, state)
            for operator in self.measurement_.M_
        ])
        raw_diagonal = np.column_stack([
            np.sum(state * state * np.diag(operator)[None, :], axis=1)
            for operator in self.measurement_.M_
        ])
        normalizer = np.maximum(raw_total.sum(axis=1, keepdims=True), 1e-12)
        diagonal_normalizer = np.maximum(
            raw_diagonal.sum(axis=1, keepdims=True), 1e-12
        )
        return {
            "probability": raw_total / normalizer,
            "diagonal": raw_diagonal / normalizer,
            "diagonal_only_probability": raw_diagonal / diagonal_normalizer,
            "cross_orbital": (raw_total - raw_diagonal) / normalizer,
        }

    @property
    def eigenvalues_(self):
        if self.core_ is None:
            return self.selection_eigenvalues_
        return self.core_.eigensystem.eigenvalues
