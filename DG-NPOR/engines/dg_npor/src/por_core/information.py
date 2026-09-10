"""
PDF Sections 1.2, 1.3 and 1.8:
Pointwise predictive information and predictive-information potential.

J(x) = D_KL(P(Y|x) || P(Y))
V_PI(x) = B_pi - J(x),  B_pi = log(1 / min_c pi_c)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse

EPS = 1e-12


def class_priors(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training set.")
    return counts / counts.sum()


def leave_one_out_kernel_posterior(
    W: sparse.spmatrix,
    y: np.ndarray,
    priors: np.ndarray,
    lambda0: float = 1.0,
) -> np.ndarray:
    """
    PDF Eq. (finite-sample posterior):
      eta_hat_ic =
        [sum_{j != i} w_ij 1(y_j=c) + lambda0*pi_hat_c]
        / [sum_{j != i} w_ij + lambda0]

    W must have zero diagonal, so leave-one-out is automatic.
    """
    W = sparse.csr_matrix(W, dtype=float)
    y = np.asarray(y, dtype=int)
    priors = np.asarray(priors, dtype=float)

    C = len(priors)
    one_hot = np.eye(C, dtype=float)[y]

    weighted_counts = W @ one_hot
    degree = np.asarray(W.sum(axis=1)).ravel()

    eta = (
        weighted_counts + float(lambda0) * priors[None, :]
    ) / (
        degree[:, None] + float(lambda0)
    )

    eta = np.clip(eta, EPS, 1.0)
    eta /= eta.sum(axis=1, keepdims=True)
    return eta


def posterior_from_affinities(
    affinities: np.ndarray,
    y_train: np.ndarray,
    priors: np.ndarray,
    lambda0: float = 1.0,
) -> np.ndarray:
    """
    PDF Section 1.10 for unseen x:
      eta_hat_c(x) =
        [sum_j w_j(x) 1(y_j=c) + lambda0*pi_hat_c]
        / [d(x) + lambda0]
    """
    if sparse.issparse(affinities):
        A = sparse.csr_matrix(affinities, dtype=float)
    else:
        A = np.asarray(affinities, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    priors = np.asarray(priors, dtype=float)

    one_hot = np.eye(len(priors), dtype=float)[y_train]
    weighted_counts = A @ one_hot
    degree = np.asarray(A.sum(axis=1)).ravel()

    eta = (
        weighted_counts + float(lambda0) * priors[None, :]
    ) / (
        degree[:, None] + float(lambda0)
    )
    eta = np.clip(eta, EPS, 1.0)
    eta /= eta.sum(axis=1, keepdims=True)
    return eta


def pointwise_predictive_information(
    posterior: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """
    PDF Section 1.2:
      J(x) = sum_c eta_c(x) log[eta_c(x)/pi_c].
    """
    eta = np.asarray(posterior, dtype=float)
    pi = np.asarray(priors, dtype=float)

    return np.sum(
        eta * (
            np.log(np.maximum(eta, EPS))
            - np.log(np.maximum(pi[None, :], EPS))
        ),
        axis=1,
    )


@dataclass(frozen=True)
class PredictivePotential:
    J: np.ndarray
    V: np.ndarray
    B_pi: float


def predictive_information_potential(
    posterior: np.ndarray,
    priors: np.ndarray,
) -> PredictivePotential:
    """
    PDF Section 1.3:
      B_pi = log(1/min_c pi_c)
      V_PI(x) = B_pi - J(x)
    """
    J = pointwise_predictive_information(posterior, priors)
    B_pi = float(np.log(1.0 / np.min(priors)))
    V = B_pi - J

    # Only numerical roundoff is clipped; analytically V >= 0.
    V = np.maximum(V, 0.0)
    return PredictivePotential(J=J, V=V, B_pi=B_pi)
