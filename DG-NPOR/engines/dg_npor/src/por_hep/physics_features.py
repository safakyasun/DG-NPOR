"""Physics-consistent feature map for the ATLAS HiggsML table.

The map is deliberately deterministic and contains no trainable classifier.
It respects the periodicity of azimuthal angles, preserves undefined detector
quantities with missingness indicators, and constructs a small set of
kinematic invariants from the primary objects.  Imputation and scaling are fit
on the training split only.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


EPS = 1e-12


def wrap_delta_phi(phi_a, phi_b):
    """Return the signed azimuthal separation in [-pi, pi]."""
    return np.arctan2(np.sin(phi_a - phi_b), np.cos(phi_a - phi_b))


class PhysicsInvariantFeatureMap:
    """Construct a train-only, topology-aware HEP representation.

    Parameters
    ----------
    feature_names:
        Names of the 30 HiggsML physics variables in input-column order.
    add_missing_indicators:
        Preserve whether a detector/derived quantity was undefined.
    add_log_features:
        Add log1p transforms for non-negative momentum, mass and energy-like
        variables.  The original variable is retained.
    add_kinematic_features:
        Add periodic angular separations, delta-R, transverse masses and
        massless two-object invariant masses when the required columns exist.
    """

    OBJECTS = {
        "tau": ("PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi"),
        "lep": ("PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi"),
        "jet1": (
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
        ),
        "jet2": (
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
        ),
    }

    PAIRS = (
        ("tau", "lep"),
        ("tau", "jet1"),
        ("lep", "jet1"),
        ("jet1", "jet2"),
    )

    def __init__(
        self,
        feature_names: Sequence[str],
        missing_sentinel: float = -999.0,
        add_missing_indicators: bool = True,
        add_log_features: bool = True,
        add_kinematic_features: bool = True,
        drop_raw_phi: bool = True,
        clip: float = 10.0,
    ):
        self.feature_names = list(feature_names)
        self.missing_sentinel = float(missing_sentinel)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.add_log_features = bool(add_log_features)
        self.add_kinematic_features = bool(add_kinematic_features)
        self.drop_raw_phi = bool(drop_raw_phi)
        self.clip = float(clip)

    def _as_named(self, X) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Input dimension does not match feature_names.")
        X = X.copy()
        X[np.isclose(X, self.missing_sentinel)] = np.nan
        X[~np.isfinite(X)] = np.nan
        return {name: X[:, j] for j, name in enumerate(self.feature_names)}

    @staticmethod
    def _valid(*values):
        mask = np.ones(len(values[0]), dtype=bool)
        for value in values:
            mask &= np.isfinite(value)
        return mask

    @staticmethod
    def _safe_sqrt(value):
        return np.sqrt(np.maximum(value, 0.0))

    def _append(self, columns, names, name, value):
        value = np.asarray(value, dtype=float)
        columns.append(value)
        names.append(str(name))

    def _raw_and_periodic(self, named, columns, names):
        for name in self.feature_names:
            value = named[name]
            is_phi = name.endswith("_phi")
            if not (is_phi and self.drop_raw_phi):
                self._append(columns, names, name, value)

            if is_phi:
                self._append(columns, names, name + "__sin", np.sin(value))
                self._append(columns, names, name + "__cos", np.cos(value))

            if self.add_log_features and any(
                token in name for token in ("_pt", "_mass", "_sumet")
            ):
                valid = value[np.isfinite(value)]
                if valid.size and np.min(valid) >= 0.0:
                    self._append(
                        columns,
                        names,
                        name + "__log1p",
                        np.log1p(np.maximum(value, 0.0)),
                    )

    def _angular_pair(self, named, key_a, key_b, columns, names):
        pt_a, eta_a, phi_a = (named[n] for n in self.OBJECTS[key_a])
        pt_b, eta_b, phi_b = (named[n] for n in self.OBJECTS[key_b])
        valid = self._valid(pt_a, eta_a, phi_a, pt_b, eta_b, phi_b)

        dphi = np.full(len(pt_a), np.nan)
        deta = np.full(len(pt_a), np.nan)
        dr = np.full(len(pt_a), np.nan)
        mass = np.full(len(pt_a), np.nan)
        ratio = np.full(len(pt_a), np.nan)

        dphi[valid] = wrap_delta_phi(phi_a[valid], phi_b[valid])
        deta[valid] = eta_a[valid] - eta_b[valid]
        dr[valid] = np.sqrt(deta[valid] ** 2 + dphi[valid] ** 2)
        mass2 = 2.0 * pt_a[valid] * pt_b[valid] * (
            np.cosh(np.clip(deta[valid], -20.0, 20.0))
            - np.cos(dphi[valid])
        )
        mass[valid] = self._safe_sqrt(mass2)
        ratio[valid] = np.log(
            np.maximum(pt_a[valid], EPS) / np.maximum(pt_b[valid], EPS)
        )

        prefix = "%s_%s" % (key_a, key_b)
        self._append(columns, names, prefix + "__abs_dphi", np.abs(dphi))
        self._append(columns, names, prefix + "__cos_dphi", np.cos(dphi))
        self._append(columns, names, prefix + "__deta", deta)
        self._append(columns, names, prefix + "__delta_r", dr)
        self._append(columns, names, prefix + "__massless_mass", mass)
        self._append(columns, names, prefix + "__log_pt_ratio", ratio)

    def _met_pair(self, named, key, columns, names):
        pt, _, phi = (named[n] for n in self.OBJECTS[key])
        if "PRI_met" not in named or "PRI_met_phi" not in named:
            return
        met = named["PRI_met"]
        met_phi = named["PRI_met_phi"]
        valid = self._valid(pt, phi, met, met_phi)
        dphi = np.full(len(pt), np.nan)
        mt = np.full(len(pt), np.nan)
        dphi[valid] = wrap_delta_phi(phi[valid], met_phi[valid])
        mt2 = 2.0 * pt[valid] * met[valid] * (1.0 - np.cos(dphi[valid]))
        mt[valid] = self._safe_sqrt(mt2)
        self._append(columns, names, key + "_met__abs_dphi", np.abs(dphi))
        self._append(columns, names, key + "_met__cos_dphi", np.cos(dphi))
        self._append(columns, names, key + "_met__transverse_mass", mt)

    def _topology(self, named, columns, names):
        if "PRI_jet_num" not in named:
            return
        jet_num = named["PRI_jet_num"]
        for category, mask in (
            ("0", jet_num < 0.5),
            ("1", (jet_num >= 0.5) & (jet_num < 1.5)),
            ("2plus", jet_num >= 1.5),
        ):
            self._append(
                columns,
                names,
                "jet_topology__" + category,
                mask.astype(float),
            )

    def _engineer_base(self, X) -> Tuple[np.ndarray, List[str]]:
        named = self._as_named(X)
        columns: List[np.ndarray] = []
        names: List[str] = []
        self._raw_and_periodic(named, columns, names)

        if self.add_kinematic_features:
            for key_a, key_b in self.PAIRS:
                required = self.OBJECTS[key_a] + self.OBJECTS[key_b]
                if all(name in named for name in required):
                    self._angular_pair(
                        named, key_a, key_b, columns, names
                    )
            for key in ("tau", "lep"):
                if all(name in named for name in self.OBJECTS[key]):
                    self._met_pair(named, key, columns, names)
            self._topology(named, columns, names)

        return np.column_stack(columns), names

    def fit(self, X):
        Z, names = self._engineer_base(X)
        self.base_output_feature_names_ = list(names)
        if self.add_missing_indicators:
            self.missing_indicator_indices_ = np.flatnonzero(
                np.any(~np.isfinite(Z), axis=0)
            )
        else:
            self.missing_indicator_indices_ = np.empty(0, dtype=int)
        if self.missing_indicator_indices_.size:
            missing = (~np.isfinite(Z)).astype(float)
            Z = np.hstack([Z, missing[:, self.missing_indicator_indices_]])
            names = list(names) + [
                names[j] + "__missing"
                for j in self.missing_indicator_indices_
            ]
        self.output_feature_names_ = list(names)
        self.imputer_ = SimpleImputer(
            strategy="median", keep_empty_features=True
        )
        Zi = self.imputer_.fit_transform(Z)
        self.scaler_ = RobustScaler(quantile_range=(10.0, 90.0))
        self.scaler_.fit(Zi)
        return self

    def transform(self, X):
        Z, names = self._engineer_base(X)
        if names != self.base_output_feature_names_:
            raise RuntimeError("Base feature-map output changed after fit.")
        if self.missing_indicator_indices_.size:
            missing = (~np.isfinite(Z)).astype(float)
            Z = np.hstack([Z, missing[:, self.missing_indicator_indices_]])
            names = list(names) + [
                names[j] + "__missing"
                for j in self.missing_indicator_indices_
            ]
        if names != self.output_feature_names_:
            raise RuntimeError("Feature-map output changed between fit and transform.")
        Zi = self.imputer_.transform(Z)
        out = self.scaler_.transform(Zi)
        if self.clip > 0:
            out = np.clip(out, -self.clip, self.clip)
        return np.asarray(out, dtype=float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def jet_categories(self, X):
        named = self._as_named(X)
        if "PRI_jet_num" not in named:
            return np.zeros(len(X), dtype=int)
        jet_num = np.nan_to_num(named["PRI_jet_num"], nan=0.0)
        return np.where(jet_num < 0.5, 0, np.where(jet_num < 1.5, 1, 2))
