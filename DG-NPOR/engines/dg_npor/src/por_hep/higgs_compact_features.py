"""Compact, azimuth-invariant input map for the ATLAS HiggsML table.

This module is an input adapter.  It does not change the POR operator,
orbital construction, automatic dimension rule, or PSD measurement in
``por_core``.  The HiggsML table already contains curated ``DER_*`` physics
variables, so the adapter avoids constructing a second, highly redundant
bank of masses, logarithms, pairwise distances, and topology features.
"""

from typing import Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class CompactATLASPhysicsFeatureMap:
    """Encode HiggsML events with a compact, global-azimuth-invariant map.

    All non-azimuthal HiggsML columns are retained once.  The unobservable
    common azimuth is removed by expressing the four visible-object angles
    relative to missing transverse momentum, with sine/cosine coordinates:

    ``(sin(phi_object - phi_MET), cos(phi_object - phi_MET))``.

    Columns that contain the HiggsML ``-999`` sentinel in the training split
    receive one binary missingness flag each.  Median imputation and robust
    scaling are fitted on the training split only.  Missingness flags are not
    scaled, so they remain exactly zero or one.
    """

    REFERENCE_PHI = "PRI_met_phi"
    OBJECT_PHI = (
        "PRI_tau_phi",
        "PRI_lep_phi",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_phi",
    )

    def __init__(
        self,
        feature_names: Sequence[str],
        missing_sentinel: float = -999.0,
        clip: float = 10.0,
    ):
        self.feature_names = list(feature_names)
        self.missing_sentinel = float(missing_sentinel)
        self.clip = float(clip)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique.")
        required = set(self.OBJECT_PHI + (self.REFERENCE_PHI,))
        missing = sorted(required.difference(self.feature_names))
        if missing:
            raise ValueError(
                "Compact ATLAS map is missing required phi columns: %s"
                % ", ".join(missing)
            )
        self.non_phi_indices_ = [
            index
            for index, name in enumerate(self.feature_names)
            if not name.endswith("_phi")
        ]
        self.phi_index_ = {
            name: self.feature_names.index(name)
            for name in self.OBJECT_PHI + (self.REFERENCE_PHI,)
        }

    def _clean_raw(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Input dimension does not match feature_names.")
        cleaned = X.copy()
        cleaned[np.isclose(cleaned, self.missing_sentinel)] = np.nan
        cleaned[~np.isfinite(cleaned)] = np.nan
        return cleaned

    def _continuous_map(self, cleaned):
        blocks = [cleaned[:, self.non_phi_indices_]]
        names = [self.feature_names[index] for index in self.non_phi_indices_]
        reference = cleaned[:, self.phi_index_[self.REFERENCE_PHI]]
        for name in self.OBJECT_PHI:
            delta_phi = cleaned[:, self.phi_index_[name]] - reference
            label = name[:-4] if name.endswith("_phi") else name
            blocks.extend([
                np.sin(delta_phi)[:, None],
                np.cos(delta_phi)[:, None],
            ])
            names.extend([
                "sin_dphi_%s_met" % label,
                "cos_dphi_%s_met" % label,
            ])
        return np.column_stack(blocks), names

    def fit(self, X):
        cleaned = self._clean_raw(X)
        mapped, names = self._continuous_map(cleaned)
        self.continuous_feature_names_ = list(names)
        self.missing_original_indices_ = np.flatnonzero(
            np.any(~np.isfinite(cleaned), axis=0)
        )
        self.missing_feature_names_ = [
            "missing__%s" % self.feature_names[index]
            for index in self.missing_original_indices_
        ]
        self.imputer_ = SimpleImputer(
            strategy="median", keep_empty_features=True
        )
        imputed = self.imputer_.fit_transform(mapped)
        self.scaler_ = RobustScaler(quantile_range=(10.0, 90.0))
        self.scaler_.fit(imputed)
        self.output_feature_names_ = (
            self.continuous_feature_names_ + self.missing_feature_names_
        )
        return self

    def transform(self, X):
        if not hasattr(self, "imputer_"):
            raise RuntimeError("CompactATLASPhysicsFeatureMap must be fit first.")
        cleaned = self._clean_raw(X)
        mapped, names = self._continuous_map(cleaned)
        if names != self.continuous_feature_names_:
            raise RuntimeError("Continuous feature-map output changed after fit.")
        continuous = self.scaler_.transform(self.imputer_.transform(mapped))
        if self.clip > 0:
            continuous = np.clip(continuous, -self.clip, self.clip)
        if self.missing_original_indices_.size:
            missing = (~np.isfinite(
                cleaned[:, self.missing_original_indices_]
            )).astype(float)
            result = np.hstack([continuous, missing])
        else:
            result = continuous
        if not np.isfinite(result).all():
            raise RuntimeError("Compact ATLAS preprocessing produced non-finite values.")
        return np.asarray(result, dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def summary(self):
        if not hasattr(self, "output_feature_names_"):
            raise RuntimeError("CompactATLASPhysicsFeatureMap must be fit first.")
        return {
            "name": "atlas_higgs_compact_azimuth_invariant_v1",
            "labels_used": False,
            "fit_scope": "training_only_imputation_and_robust_scaling",
            "raw_dimension": int(len(self.feature_names)),
            "retained_non_phi_dimension": int(len(self.non_phi_indices_)),
            "relative_phi_dimension": int(2 * len(self.OBJECT_PHI)),
            "continuous_dimension": int(len(self.continuous_feature_names_)),
            "missing_indicator_dimension": int(
                len(self.missing_original_indices_)
            ),
            "output_dimension": int(len(self.output_feature_names_)),
            "absolute_phi_retained": False,
            "global_azimuth_rotation_invariant": True,
            "redundant_kinematic_expansion": False,
            "missing_indicator_source_columns": [
                self.feature_names[index]
                for index in self.missing_original_indices_
            ],
        }
