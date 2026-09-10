"""Transparent logistic probes for validation and post-selection audit.

These probes never define the orbital dimension and are never the final POR
readout.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def make_probabilistic_probe(C=1.0, max_iter=3000, random_state=42):
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            C=float(C), solver="lbfgs", max_iter=int(max_iter),
            random_state=int(random_state),
        )),
    ])


def fit_point_losses(X_fit, y_fit, X_val, y_val, probe):
    probe.fit(np.asarray(X_fit, dtype=float), np.asarray(y_fit, dtype=int))
    probability = np.asarray(
        probe.predict_proba(np.asarray(X_val, dtype=float)), dtype=float
    )
    classes = np.asarray(probe.named_steps["logistic"].classes_, dtype=int)
    aligned = np.full((len(X_val), 2), EPS, dtype=float)
    aligned[:, classes] = probability
    aligned /= aligned.sum(axis=1, keepdims=True)
    y_val = np.asarray(y_val, dtype=int)
    losses = -np.log(np.maximum(aligned[np.arange(len(y_val)), y_val], EPS))
    return losses, aligned
