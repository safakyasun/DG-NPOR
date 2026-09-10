"""Neural geometry map used only before the POR operator.

The auxiliary binary output is used to train a smooth task-aware metric on a
training-only split.  Its probability is never fused with, calibrated into,
or substituted for the final structured POR probability.  The map h_theta is
the penultimate hidden representation and has the same dimension as its
physics-consistent input, so it is not the claimed compact bottleneck.
"""

from copy import deepcopy

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class NeuralGeometryMap:
    """Training-only MLP metric with an auditable manual early-stop loop."""

    def __init__(
        self,
        max_epochs=250,
        patience=25,
        validation_fraction=0.15,
        learning_rate=1e-3,
        alpha=1e-4,
        batch_size=256,
        architecture="standard",
        random_state=42,
    ):
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.architecture = str(architecture)
        self.random_state = int(random_state)

    def _validate(self):
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive.")
        if not 0.05 <= self.validation_fraction <= 0.40:
            raise ValueError("validation_fraction must lie in [0.05, 0.40].")
        if self.learning_rate <= 0 or self.alpha < 0:
            raise ValueError("Invalid neural-geometry regularization settings.")
        if self.architecture not in {"shallow", "standard", "deep", "tapered"}:
            raise ValueError(
                "architecture must be shallow, standard, deep, or tapered."
            )

    def _hidden_widths(self, dimension):
        """Return a bounded, input-dimension-preserving h_theta architecture."""
        dimension = int(dimension)
        double = min(256, max(16, 2 * dimension))
        triple = min(256, max(24, 3 * dimension))
        if self.architecture == "shallow":
            return (dimension,)
        if self.architecture == "standard":
            return (double, dimension)
        if self.architecture == "deep":
            return (double, double, dimension)
        return (triple, double, dimension)

    @staticmethod
    def _tanh_hidden(X, coefs, intercepts):
        hidden = np.asarray(X, dtype=float)
        for weight, bias in zip(coefs[:-1], intercepts[:-1]):
            hidden = np.tanh(hidden @ weight + bias)
        return hidden

    def fit(self, X, y):
        self._validate()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y) or X.shape[1] < 2:
            raise ValueError("Neural geometry requires a finite 2D feature table.")
        if not np.isfinite(X).all() or set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("Neural geometry expects finite inputs and labels 0/1.")

        fit_index, validation_index = train_test_split(
            np.arange(len(X), dtype=int),
            test_size=self.validation_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        dimension = int(X.shape[1])
        widths = self._hidden_widths(dimension)
        effective_batch = min(self.batch_size, len(fit_index))
        self.network_ = MLPClassifier(
            hidden_layer_sizes=widths,
            activation="tanh",
            solver="adam",
            alpha=self.alpha,
            batch_size=effective_batch,
            learning_rate_init=self.learning_rate,
            max_iter=1,
            shuffle=True,
            warm_start=False,
            random_state=self.random_state,
        )

        best = None
        best_loss = np.inf
        stale = 0
        history = []
        for epoch in range(1, self.max_epochs + 1):
            self.network_.partial_fit(
                X[fit_index], y[fit_index], classes=np.array([0, 1])
            )
            probability = self.network_.predict_proba(X[validation_index])
            validation_loss = float(
                log_loss(y[validation_index], probability, labels=[0, 1])
            )
            validation_auc = float(
                roc_auc_score(y[validation_index], probability[:, 1])
            )
            history.append({
                "epoch": int(epoch),
                "validation_log_loss": validation_loss,
                "validation_auc": validation_auc,
            })
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best = (
                    deepcopy(self.network_.coefs_),
                    deepcopy(self.network_.intercepts_),
                    epoch,
                    validation_auc,
                )
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best is None:
            raise RuntimeError("Neural geometry did not produce a finite state.")
        self.network_.coefs_ = best[0]
        self.network_.intercepts_ = best[1]
        self.best_epoch_ = int(best[2])
        self.validation_auc_ = float(best[3])
        self.validation_log_loss_ = float(best_loss)
        self.history_ = history
        self.fit_indices_ = np.asarray(fit_index, dtype=int)
        self.validation_indices_ = np.asarray(validation_index, dtype=int)
        self.input_dimension_ = dimension
        self.output_dimension_ = dimension
        self.hidden_widths_ = tuple(int(value) for value in widths)
        hidden = self._tanh_hidden(
            X, self.network_.coefs_, self.network_.intercepts_
        )
        self.output_scaler_ = StandardScaler().fit(hidden)
        return self

    def transform(self, X):
        if not hasattr(self, "network_"):
            raise RuntimeError("NeuralGeometryMap must be fit before transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dimension_:
            raise ValueError("Neural geometry input dimension changed.")
        hidden = self._tanh_hidden(
            X, self.network_.coefs_, self.network_.intercepts_
        )
        result = self.output_scaler_.transform(hidden)
        if not np.isfinite(result).all():
            raise RuntimeError("Neural geometry produced non-finite values.")
        return np.asarray(result, dtype=float)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def summary(self):
        return {
            "role": "training_only_geometry_map_h_theta",
            "architecture": self.architecture,
            "architecture_rule": "bounded data-selected tanh family; final hidden width equals d_physics",
            "hidden_widths": list(self.hidden_widths_),
            "input_dimension": int(self.input_dimension_),
            "geometry_dimension": int(self.output_dimension_),
            "geometry_is_claimed_compact_embedding": False,
            "best_epoch": int(self.best_epoch_),
            "validation_log_loss": float(self.validation_log_loss_),
            "validation_auc": float(self.validation_auc_),
            "auxiliary_probability_used_in_final_prediction": False,
            "auxiliary_probability_fusion": False,
        }
