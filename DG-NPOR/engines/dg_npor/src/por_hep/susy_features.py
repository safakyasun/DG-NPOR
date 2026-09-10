"""Training-only physical representation for the UCI SUSY table."""

from typing import Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


SUSY_FEATURE_NAMES = [
    "lep1_pt", "lep1_eta", "lep1_phi",
    "lep2_pt", "lep2_eta", "lep2_phi",
    "missing_energy", "missing_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R",
    "M_Delta_R", "dPhi_r_b", "cos_theta_r1",
]


class SUSYPhysicsFeatureMap:
    """Encode SUSY low/high-level features without using labels.

    The UCI SUSY table already contains ten high-level physics variables, so
    they are retained directly.  The three azimuthal angles are replaced by
    sine/cosine coordinates and their pairwise angular separations are added
    as sine/cosine pairs.  Median imputation and robust scaling are fit only
    on the metric/train split.
    """

    PHI_INDICES = (2, 5, 7)
    PHI_PAIRS = (
        (2, 5, "lep1_lep2"),
        (2, 7, "lep1_met"),
        (5, 7, "lep2_met"),
    )

    def __init__(
        self,
        feature_names: Sequence[str] = SUSY_FEATURE_NAMES,
        missing_sentinel: float = -999.0,
        clip: float = 10.0,
    ):
        self.feature_names = list(feature_names)
        self.missing_sentinel = float(missing_sentinel)
        self.clip = float(clip)
        if len(self.feature_names) != 18:
            raise ValueError("SUSY feature map expects exactly 18 variables.")

    def _map(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim != 2 or X.shape[1] != 18:
            raise ValueError("SUSY input must have shape (n, 18).")
        X[np.isclose(X, self.missing_sentinel)] = np.nan
        linear_indices = [i for i in range(18) if i not in self.PHI_INDICES]
        blocks = [X[:, linear_indices]]
        names = [self.feature_names[i] for i in linear_indices]

        for index in self.PHI_INDICES:
            angle = X[:, index]
            blocks.extend([np.sin(angle)[:, None], np.cos(angle)[:, None]])
            names.extend([
                "sin_%s" % self.feature_names[index],
                "cos_%s" % self.feature_names[index],
            ])

        for first, second, label in self.PHI_PAIRS:
            delta = X[:, first] - X[:, second]
            blocks.extend([np.sin(delta)[:, None], np.cos(delta)[:, None]])
            names.extend(["sin_dphi_%s" % label, "cos_dphi_%s" % label])
        return np.column_stack(blocks), names

    def fit(self, X):
        mapped, names = self._map(X)
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        imputed = self.imputer_.fit_transform(mapped)
        self.scaler_ = RobustScaler(quantile_range=(10.0, 90.0)).fit(imputed)
        indicator_names = [
            "missing_%s" % names[index]
            for index in self.imputer_.indicator_.features_
        ]
        self.output_feature_names_ = names + indicator_names
        return self

    def transform(self, X):
        if not hasattr(self, "imputer_"):
            raise RuntimeError("SUSYPhysicsFeatureMap must be fit first.")
        mapped, _ = self._map(X)
        result = self.scaler_.transform(self.imputer_.transform(mapped))
        result = np.clip(result, -self.clip, self.clip)
        if not np.isfinite(result).all():
            raise RuntimeError("SUSY preprocessing produced non-finite values.")
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def summary(self):
        if not hasattr(self, "output_feature_names_"):
            raise RuntimeError("SUSYPhysicsFeatureMap must be fit first.")
        return {
            "name": "susy_physics_azimuth_periodic_v1",
            "labels_used": False,
            "fit_scope": "base_training_role_only",
            "raw_feature_dimension": int(len(self.feature_names)),
            "output_dimension": int(len(self.output_feature_names_)),
            "construction": (
                "15 non-phi supplied variables + sin/cos of 3 phi angles + "
                "sin/cos of 3 pairwise delta-phi angles"
            ),
            "imputation": "training_median",
            "scaling": "training_robust_10_90_quantiles",
            "PCA": False,
            "user_selected_bottleneck": False,
        }
