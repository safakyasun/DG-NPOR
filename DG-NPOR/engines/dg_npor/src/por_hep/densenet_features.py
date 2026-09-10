"""Training-only preprocessing for frozen DenseNet201 GAP embeddings."""

from typing import Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DenseNet201DirectFeatureMap:
    """Prepare DenseNet201 features without a user-chosen bottleneck.

    Median imputation, exact zero-variance removal, and standardization are fit
    on the base/training role only.  No label and no PCA component count enters
    this adapter.  Automatic compactness remains the responsibility of the POR
    candidate-bank and dJ/dgate orbital rules.
    """

    def __init__(self, feature_names: Sequence[str], variance_tolerance=0.0):
        self.feature_names = list(feature_names)
        self.variance_tolerance = float(variance_tolerance)
        if len(self.feature_names) < 2:
            raise ValueError("DenseNet feature map requires at least two columns.")
        if self.variance_tolerance < 0:
            raise ValueError("variance_tolerance cannot be negative.")

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("DenseNet input dimension does not match feature_names.")
        return X

    def fit(self, X):
        X = self._validate_input(X)
        self.imputer_ = SimpleImputer(
            strategy="median", keep_empty_features=True
        )
        imputed = self.imputer_.fit_transform(X)
        variance = np.var(imputed, axis=0)
        self.retained_indices_ = np.flatnonzero(
            np.isfinite(variance) & (variance > self.variance_tolerance)
        )
        self.removed_indices_ = np.setdiff1d(
            np.arange(len(self.feature_names), dtype=int),
            self.retained_indices_,
        )
        if len(self.retained_indices_) < 2:
            raise ValueError("Fewer than two non-constant DenseNet channels remain.")
        retained = imputed[:, self.retained_indices_]
        self.scaler_ = StandardScaler().fit(retained)
        self.output_feature_names_ = [
            self.feature_names[index] for index in self.retained_indices_
        ]
        return self

    def transform(self, X):
        if not hasattr(self, "imputer_"):
            raise RuntimeError("DenseNet201DirectFeatureMap must be fit first.")
        X = self._validate_input(X)
        imputed = self.imputer_.transform(X)
        result = self.scaler_.transform(imputed[:, self.retained_indices_])
        if not np.isfinite(result).all():
            raise RuntimeError("DenseNet preprocessing produced non-finite values.")
        return np.asarray(result, dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def summary(self):
        if not hasattr(self, "output_feature_names_"):
            raise RuntimeError("DenseNet201DirectFeatureMap must be fit first.")
        return {
            "name": "densenet201_gap_direct_standardized_v1",
            "source": "frozen_ImageNet_DenseNet201_global_average_pooling",
            "labels_used": False,
            "fit_scope": "base_training_role_only",
            "raw_feature_dimension": int(len(self.feature_names)),
            "retained_nonconstant_dimension": int(len(self.retained_indices_)),
            "removed_constant_dimension": int(len(self.removed_indices_)),
            "removed_feature_names": [
                self.feature_names[index] for index in self.removed_indices_
            ],
            "imputation": "training_median",
            "scaling": "training_standardization",
            "PCA": False,
            "user_selected_bottleneck": False,
        }
