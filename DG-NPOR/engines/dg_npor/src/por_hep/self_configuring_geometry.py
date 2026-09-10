"""Physics-constrained, sparse self-configuring track geometry.

The legal branch library is fixed by reconstructed track physics.  Data may
switch off non-residual branches through hard-concrete-style gates, while an
outer development-only search selects width, depth, attention heads, and the
training-only h_theta family.  The final POR operator and measurement are not
implemented here and remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .atlas_jetset import RECONSTRUCTED_TRACK_INPUT_FIELDS
from .structured_tracks import StructuredTrackGeometry, StructuredTrackScaler


class PhysicsConstrainedGatedTrackGeometry(StructuredTrackGeometry):
    """Hybrid geometry with sparse branch gates and configurable NN capacity."""

    def __init__(
        self,
        hidden_dimension=24,
        geometry_dimension=48,
        geometry_depth=1,
        attention_heads=1,
        max_tracks=20,
        graph_neighbors=6,
        sequential_steps=3,
        parallel_temperature=2.0,
        parallel_floor=0.02,
        gate_l0=2e-4,
        gate_temperature=0.67,
        gate_threshold=0.50,
        min_nonresidual_branches=1,
        learning_rate=2e-3,
        l2=1e-4,
        batch_size=512,
        max_epochs=12,
        patience=3,
        random_state=42,
    ):
        super().__init__(
            representation="hybrid",
            hidden_dimension=hidden_dimension,
            geometry_dimension=geometry_dimension,
            max_tracks=max_tracks,
            graph_neighbors=graph_neighbors,
            sequential_steps=sequential_steps,
            parallel_temperature=parallel_temperature,
            parallel_floor=parallel_floor,
            learning_rate=learning_rate,
            l2=l2,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.geometry_depth = int(geometry_depth)
        self.attention_heads = int(attention_heads)
        self.gate_l0 = float(gate_l0)
        self.gate_temperature = float(gate_temperature)
        self.gate_threshold = float(gate_threshold)
        self.min_nonresidual_branches = int(min_nonresidual_branches)
        if self.geometry_depth not in {1, 2, 3}:
            raise ValueError("geometry_depth must be 1, 2, or 3.")
        if self.attention_heads not in {1, 2, 4}:
            raise ValueError("attention_heads must be 1, 2, or 4.")
        if self.gate_l0 < 0 or self.gate_temperature <= 0:
            raise ValueError("Invalid sparse-gate regularization settings.")
        if not 0.0 < self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must lie in (0, 1).")

    @property
    def branch_names(self):
        return tuple(
            ["H"]
            + ["A^%d H" % value for value in range(1, self.sequential_steps + 1)]
            + ["G H", "H-G H"]
        )

    def _branch_values(self, tracks, mask):
        local = self._local_adjacency(tracks, mask)
        sequential = self._sequential_messages(local, tracks)
        broadcast = np.matmul(self._parallel_adjacency(tracks, mask), tracks)
        return [tracks] + sequential + [broadcast, tracks - broadcast]

    @staticmethod
    def _sigmoid_scalar(values):
        values = np.clip(np.asarray(values, dtype=np.float32), -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _soft_gate_state(self):
        logits = self.params_["gate_logits"]
        lower, upper = -0.1, 1.1
        sigmoid = self._sigmoid_scalar(logits / self.gate_temperature)
        stretched = sigmoid * (upper - lower) + lower
        gates = np.clip(stretched, 0.0, 1.0)
        derivative = (
            (upper - lower)
            * sigmoid
            * (1.0 - sigmoid)
            / self.gate_temperature
        )
        derivative *= (stretched > 0.0) & (stretched < 1.0)
        l0_logit = logits - self.gate_temperature * np.log(-lower / upper)
        expected_l0 = self._sigmoid_scalar(l0_logit)
        expected_derivative = expected_l0 * (1.0 - expected_l0)
        return gates, derivative, expected_l0, expected_derivative

    def _gate_values(self):
        if getattr(self, "structure_finalized_", False):
            return np.asarray(self.hard_gate_values_, dtype=np.float32)
        soft, _, _, _ = self._soft_gate_state()
        return np.concatenate([np.ones(1, dtype=np.float32), soft])

    def _node_input_and_branches(self, tracks, mask):
        branches = self._branch_values(tracks, mask)
        gates = self._gate_values()
        gated = [branch * gates[index] for index, branch in enumerate(branches)]
        return np.concatenate(gated, axis=2).astype(np.float32), branches, gates

    def _node_input(self, tracks, mask):
        if not hasattr(self, "params_"):
            return np.concatenate(self._branch_values(tracks, mask), axis=2).astype(
                np.float32
            )
        return self._node_input_and_branches(tracks, mask)[0]

    def _initialize(self, node_dimension):
        rng = np.random.RandomState(self.random_state)

        def weight(rows, columns):
            limit = np.sqrt(6.0 / float(rows + columns))
            return rng.uniform(-limit, limit, size=(rows, columns)).astype(np.float32)

        pooled = (self.attention_heads + 1) * self.hidden_dimension + 4
        params = {
            "Wn": weight(node_dimension, self.hidden_dimension),
            "bn": np.zeros(self.hidden_dimension, dtype=np.float32),
            "wa": weight(self.hidden_dimension, self.attention_heads),
            "wo": weight(self.geometry_dimension, 1).ravel(),
            "bo": np.zeros(1, dtype=np.float32),
            # Start open but below the hard-concrete upper clip so both the
            # predictive loss and the sparsity term can move every branch.
            "gate_logits": np.full(len(self.branch_names) - 1, 1.0, dtype=np.float32),
        }
        for layer in range(self.geometry_depth):
            incoming = pooled if layer == 0 else self.geometry_dimension
            params["Wg_%d" % layer] = weight(incoming, self.geometry_dimension)
            params["bg_%d" % layer] = np.zeros(
                self.geometry_dimension, dtype=np.float32
            )
        self.params_ = params
        self.structure_finalized_ = False

    def _forward(self, tracks, mask, jets, cache=False):
        node_input, branches, gates = self._node_input_and_branches(tracks, mask)
        pre_node = np.matmul(node_input, self.params_["Wn"]) + self.params_["bn"]
        node_raw = np.tanh(pre_node).astype(np.float32)
        node = node_raw * mask[:, :, None]
        root_h = np.sqrt(float(self.hidden_dimension))
        attention_logits = np.matmul(node, self.params_["wa"]) / root_h
        attention_logits = np.where(mask[:, :, None], attention_logits, -1e9)
        shifted = attention_logits - np.max(attention_logits, axis=1, keepdims=True)
        exponent = np.exp(np.clip(shifted, -30.0, 0.0)) * mask[:, :, None]
        denominator = np.sum(exponent, axis=1, keepdims=True)
        attention = np.divide(
            exponent,
            denominator,
            out=np.zeros_like(exponent, dtype=np.float32),
            where=denominator > 0,
        ).astype(np.float32)
        pooled_attention = np.einsum("bth,btd->bhd", attention, node).reshape(
            len(node), self.attention_heads * self.hidden_dimension
        )
        counts = np.maximum(np.sum(mask, axis=1, keepdims=True), 1)
        pooled_mean = np.sum(node, axis=1) / counts
        pooled = np.concatenate([pooled_attention, pooled_mean, jets], axis=1)

        layer_inputs = []
        geometries = []
        current = pooled
        for layer in range(self.geometry_depth):
            layer_inputs.append(current)
            pre = np.matmul(current, self.params_["Wg_%d" % layer]) + self.params_[
                "bg_%d" % layer
            ]
            current = np.tanh(pre).astype(np.float32)
            geometries.append(current)
        geometry = geometries[-1]
        logits = np.matmul(geometry, self.params_["wo"]) + self.params_["bo"][0]
        probability = self._sigmoid(logits)
        if not cache:
            return geometry, probability
        return geometry, probability, {
            "node_input": node_input,
            "node_raw": node_raw,
            "node": node,
            "attention": attention,
            "counts": counts.astype(np.float32),
            "pooled": pooled,
            "layer_inputs": layer_inputs,
            "geometries": geometries,
            "branches": branches,
            "gates": gates,
        }

    def _loss_and_gradients(self, tracks, mask, jets, labels):
        geometry, probability, cache = self._forward(tracks, mask, jets, cache=True)
        labels = np.asarray(labels, dtype=np.float32)
        batch = float(len(labels))
        eps = 1e-7
        loss = -np.mean(
            labels * np.log(np.clip(probability, eps, 1.0))
            + (1.0 - labels) * np.log(np.clip(1.0 - probability, eps, 1.0))
        )
        regularized = ["Wn", "wa", "wo"] + [
            "Wg_%d" % layer for layer in range(self.geometry_depth)
        ]
        for name in regularized:
            loss += 0.5 * self.l2 * float(np.sum(self.params_[name] ** 2))
        _, gate_derivative, expected_l0, expected_derivative = self._soft_gate_state()
        loss += self.gate_l0 * float(np.sum(expected_l0))

        gradients = {}
        d_logit = (probability - labels) / batch
        gradients["wo"] = np.matmul(geometry.T, d_logit) + self.l2 * self.params_["wo"]
        gradients["bo"] = np.asarray([np.sum(d_logit)], dtype=np.float32)
        d_current = d_logit[:, None] * self.params_["wo"][None, :]

        for layer in reversed(range(self.geometry_depth)):
            value = cache["geometries"][layer]
            d_pre = d_current * (1.0 - value * value)
            name = "Wg_%d" % layer
            gradients[name] = (
                np.matmul(cache["layer_inputs"][layer].T, d_pre)
                + self.l2 * self.params_[name]
            )
            gradients["bg_%d" % layer] = np.sum(d_pre, axis=0)
            d_current = np.matmul(d_pre, self.params_[name].T)
        d_pooled = d_current

        hidden = self.hidden_dimension
        heads = self.attention_heads
        attention_width = heads * hidden
        d_attention_pool = d_pooled[:, :attention_width].reshape(
            len(labels), heads, hidden
        )
        d_mean_pool = d_pooled[:, attention_width : attention_width + hidden]
        node = cache["node"]
        attention = cache["attention"]
        d_node = np.einsum("bth,bhd->btd", attention, d_attention_pool)
        d_node += (
            d_mean_pool[:, None, :] / cache["counts"][:, :, None]
        ) * mask[:, :, None]
        d_attention = np.einsum("bhd,btd->bth", d_attention_pool, node)
        centered = d_attention - np.sum(
            d_attention * attention, axis=1, keepdims=True
        )
        d_attention_logits = attention * centered
        root_h = np.sqrt(float(hidden))
        gradients["wa"] = (
            np.einsum("btd,bth->dh", node, d_attention_logits) / root_h
            + self.l2 * self.params_["wa"]
        )
        d_node += np.einsum(
            "bth,dh->btd", d_attention_logits, self.params_["wa"]
        ) / root_h

        d_pre_node = (
            d_node
            * (1.0 - cache["node_raw"] * cache["node_raw"])
            * mask[:, :, None]
        )
        gradients["Wn"] = (
            np.einsum("bti,btj->ij", cache["node_input"], d_pre_node)
            + self.l2 * self.params_["Wn"]
        )
        gradients["bn"] = np.sum(d_pre_node, axis=(0, 1))
        d_node_input = np.matmul(d_pre_node, self.params_["Wn"].T)
        field_dimension = len(RECONSTRUCTED_TRACK_INPUT_FIELDS)
        gate_gradients = []
        for branch_index, branch in enumerate(cache["branches"][1:], start=1):
            start = branch_index * field_dimension
            stop = start + field_dimension
            d_gate = float(np.sum(d_node_input[:, :, start:stop] * branch))
            gate_gradients.append(d_gate * gate_derivative[branch_index - 1])
        gradients["gate_logits"] = np.asarray(gate_gradients, dtype=np.float32)
        gradients["gate_logits"] += self.gate_l0 * expected_derivative
        return float(loss), gradients

    def _finalize_gates(self):
        soft, _, expected_l0, _ = self._soft_gate_state()
        active = soft >= self.gate_threshold
        minimum = max(0, min(self.min_nonresidual_branches, len(active)))
        if int(np.sum(active)) < minimum:
            order = np.argsort(-soft)
            active[order[:minimum]] = True
        self.soft_branch_gate_values_ = np.concatenate([[1.0], soft]).astype(
            np.float32
        )
        self.expected_active_nonresidual_branches_ = float(np.sum(expected_l0))
        self.hard_gate_values_ = np.concatenate([[1.0], active.astype(float)]).astype(
            np.float32
        )
        self.structure_finalized_ = True

    def fit(
        self,
        source,
        train_indices,
        train_labels,
        validation_indices,
        validation_labels,
        read_batch_size=4096,
    ):
        train_indices = np.asarray(train_indices, dtype=int)
        validation_indices = np.asarray(validation_indices, dtype=int)
        train_labels = np.asarray(train_labels, dtype=int)
        validation_labels = np.asarray(validation_labels, dtype=int)
        if set(np.unique(train_labels).tolist()) != {0, 1}:
            raise ValueError("Encoder training requires both classes.")
        if set(np.unique(validation_labels).tolist()) != {0, 1}:
            raise ValueError("Encoder validation requires both classes.")

        train_raw = source.read_many(train_indices, batch_size=read_batch_size)
        validation_raw = source.read_many(validation_indices, batch_size=read_batch_size)
        self.scaler_ = StructuredTrackScaler().fit(*train_raw)
        train = self.scaler_.transform(*train_raw)
        validation = self.scaler_.transform(*validation_raw)
        node_dimension = len(self.branch_names) * len(RECONSTRUCTED_TRACK_INPUT_FIELDS)
        self._initialize(node_dimension)

        first = {name: np.zeros_like(value) for name, value in self.params_.items()}
        second = {name: np.zeros_like(value) for name, value in self.params_.items()}
        step = 0
        rng = np.random.RandomState(self.random_state + 19)
        best_params = None
        best_loss = np.inf
        best_epoch = 0
        stale = 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(train_indices))
            batch_losses = []
            for start in range(0, len(order), self.batch_size):
                chosen = order[start : start + self.batch_size]
                loss, gradients = self._loss_and_gradients(
                    train[0][chosen],
                    train[1][chosen],
                    train[2][chosen],
                    train_labels[chosen],
                )
                batch_losses.append(loss)
                step += 1
                for name in self.params_:
                    gradient = np.asarray(gradients[name], dtype=np.float32)
                    first[name] = 0.9 * first[name] + 0.1 * gradient
                    second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
                    first_hat = first[name] / (1.0 - 0.9 ** step)
                    second_hat = second[name] / (1.0 - 0.999 ** step)
                    self.params_[name] -= self.learning_rate * first_hat / (
                        np.sqrt(second_hat) + 1e-8
                    )
            _, validation_probability = self._forward(*validation, cache=False)
            record = self._validation_record(validation_labels, validation_probability)
            record.update(
                {
                    "epoch": int(epoch),
                    "training_objective": float(np.mean(batch_losses)),
                    "expected_active_nonresidual_branches": float(
                        np.sum(self._soft_gate_state()[2])
                    ),
                }
            )
            self.history_.append(record)
            if record["validation_log_loss"] < best_loss - 1e-6:
                best_loss = record["validation_log_loss"]
                best_params = deepcopy(self.params_)
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_params is None:
            raise RuntimeError("Self-configuring encoder did not produce a finite state.")
        self.params_ = best_params
        self.best_epoch_ = int(best_epoch)
        self.soft_validation_log_loss_ = float(best_loss)
        self._finalize_gates()

        train_geometry, _ = self._forward(*train, cache=False)
        self.embedding_center_ = np.mean(train_geometry, axis=0).astype(np.float32)
        self.embedding_scale_ = np.maximum(
            np.std(train_geometry, axis=0), 1e-5
        ).astype(np.float32)
        _, hard_probability = self._forward(*validation, cache=False)
        self.validation_log_loss_ = float(
            log_loss(validation_labels, hard_probability, labels=[0, 1])
        )
        self.validation_auc_ = float(roc_auc_score(validation_labels, hard_probability))
        self.node_input_dimension_ = int(node_dimension)
        self.train_rows_ = int(len(train_indices))
        self.validation_rows_ = int(len(validation_indices))
        return self

    def parameter_count(self, effective=False):
        total = int(sum(np.asarray(value).size for value in self.params_.values()))
        if not effective or not getattr(self, "structure_finalized_", False):
            return total
        field_dimension = len(RECONSTRUCTED_TRACK_INPUT_FIELDS)
        inactive = int(np.sum(self.hard_gate_values_ == 0.0))
        return int(total - inactive * field_dimension * self.hidden_dimension)

    def summary(self):
        return {
            "role": "training_only_physics_constrained_self_configuring_geometry",
            "legal_branch_library": list(self.branch_names),
            "selected_branches": [
                name
                for name, gate in zip(self.branch_names, self.hard_gate_values_)
                if gate > 0.5
            ],
            "soft_branch_gates": {
                name: float(value)
                for name, value in zip(self.branch_names, self.soft_branch_gate_values_)
            },
            "hard_branch_gates": {
                name: int(value > 0.5)
                for name, value in zip(self.branch_names, self.hard_gate_values_)
            },
            "gate_rule": "hard-concrete-style deterministic median gates; H is mandatory",
            "gate_l0_penalty": float(self.gate_l0),
            "gate_threshold": float(self.gate_threshold),
            "hidden_dimension": int(self.hidden_dimension),
            "geometry_dimension": int(self.geometry_dimension),
            "geometry_depth": int(self.geometry_depth),
            "attention_heads": int(self.attention_heads),
            "sequential_steps": int(self.sequential_steps),
            "graph_neighbors": int(self.graph_neighbors),
            "parallel_temperature": float(self.parallel_temperature),
            "parallel_floor": float(self.parallel_floor),
            "parameter_count_stored": self.parameter_count(effective=False),
            "parameter_count_effective_after_branch_pruning": self.parameter_count(
                effective=True
            ),
            "encoder_training_rows": int(self.train_rows_),
            "encoder_validation_rows": int(self.validation_rows_),
            "best_epoch": int(self.best_epoch_),
            "soft_validation_log_loss": float(self.soft_validation_log_loss_),
            "hard_validation_log_loss": float(self.validation_log_loss_),
            "validation_auc_audit_only": float(self.validation_auc_),
            "physics_constraints": {
                "permutation_invariant_pooling": True,
                "positive_row_normalized_fixed_affinities": True,
                "original_H_residual_mandatory": True,
                "padded_track_mask_respected": True,
            },
            "auxiliary_probability_used_in_final_prediction": False,
            "auxiliary_probability_fusion": False,
            "truth_track_fields_used": False,
            "GN2_or_DL1_scores_used": False,
            "GN2v01_vertexIndex_used": False,
            "scaler": self.scaler_.summary(),
        }


def dropout_stability_audit(
    encoder,
    source,
    logical_indices,
    random_state=42,
    drop_fraction=0.10,
    read_batch_size=2048,
):
    """Training/development-only score stability under reconstructed-track loss."""
    logical_indices = np.asarray(logical_indices, dtype=int)
    raw = source.read_many(logical_indices, batch_size=read_batch_size)
    tracks, mask, jets = raw
    _, original_probability = encoder.transform_arrays(tracks, mask, jets)
    rng = np.random.RandomState(int(random_state))
    perturbed_mask = mask.copy()
    removable = rng.uniform(size=mask.shape) < float(drop_fraction)
    perturbed_mask &= ~removable
    empty = np.where(np.sum(perturbed_mask, axis=1) == 0)[0]
    if len(empty):
        first_valid = np.argmax(mask[empty], axis=1)
        perturbed_mask[empty, first_valid] = True
    perturbed_tracks = tracks * perturbed_mask[:, :, None]
    _, perturbed_probability = encoder.transform_arrays(
        perturbed_tracks, perturbed_mask, jets
    )
    difference = np.abs(original_probability - perturbed_probability)
    return {
        "drop_fraction": float(drop_fraction),
        "mean_absolute_probability_shift": float(np.mean(difference)),
        "p95_absolute_probability_shift": float(np.quantile(difference, 0.95)),
    }
