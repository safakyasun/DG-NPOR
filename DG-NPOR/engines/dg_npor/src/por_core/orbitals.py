"""
PDF Sections 1.10 and 1.11:
Inductive extension and predictive orbital state.
"""

import numpy as np
from scipy import sparse
from typing import Tuple

from .information import (
    posterior_from_affinities,
    predictive_information_potential,
)

EPS = 1e-12


def extend_orbitals(
    affinities: np.ndarray,
    degree_query: np.ndarray,
    train_phi: np.ndarray,
    eigenvalues: np.ndarray,
    query_V: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    denominator_floor: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact PDF Section 1.10:

      phi_k(x) =
        alpha * sum_j [w_j(x)/d(x)] phi_k(x_j)
        / [alpha + beta V_hat(x) - epsilon_k].

    Returns:
      Phi_query : raw graph-eigenfunction coordinates.
      stable_mask : modes whose denominators never hit the floor.
    """
    A = (
        sparse.csr_matrix(affinities, dtype=float)
        if sparse.issparse(affinities)
        else np.asarray(affinities, dtype=float)
    )
    dq = np.asarray(degree_query, dtype=float)
    phi_train = np.asarray(train_phi, dtype=float)
    eig = np.asarray(eigenvalues, dtype=float)
    Vq = np.asarray(query_V, dtype=float)

    if sparse.issparse(A):
        normalized_affinity = sparse.diags(
            1.0 / np.maximum(dq, EPS)
        ) @ A
    else:
        normalized_affinity = A / np.maximum(dq[:, None], EPS)
    numerator = float(alpha) * np.asarray(normalized_affinity @ phi_train)

    denominator = (
        float(alpha)
        + float(beta) * Vq[:, None]
        - eig[None, :]
    )

    stable_mask = np.all(
        np.abs(denominator) >= float(denominator_floor),
        axis=0,
    )

    sign = np.where(denominator >= 0.0, 1.0, -1.0)
    denominator_safe = (
        sign
        * np.maximum(
            np.abs(denominator),
            float(denominator_floor),
        )
    )

    Phi_query = numerator / denominator_safe
    return Phi_query, stable_mask


def normalized_orbital_state(
    Phi: np.ndarray,
    K: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    PDF Section 1.11:

      r_K(x) = [phi_1(x), ..., phi_K(x)]^T

      a_K(x) =
        r_K(x) / sqrt(r_K(x)^T r_K(x) + epsilon).
    """
    R = np.asarray(Phi, dtype=float)[:, :int(K)]
    norm = np.sqrt(
        np.sum(R * R, axis=1, keepdims=True)
        + float(epsilon)
    )
    return R / norm


def normalized_selected_orbital_state(
    Phi: np.ndarray,
    selected_indices,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Normalize an automatically selected, possibly non-contiguous subspace.

    ``selected_indices`` is fixed using training-only information.  The result
    is the same PDF-normalized orbital state, restricted to that subspace.
    """
    Phi = np.asarray(Phi, dtype=float)
    indices = np.asarray(selected_indices, dtype=int)
    if Phi.ndim != 2 or indices.ndim != 1 or not len(indices):
        raise ValueError("A non-empty one-dimensional orbital index set is required.")
    if np.any(indices < 0) or np.any(indices >= Phi.shape[1]):
        raise ValueError("Selected orbital index lies outside the computed bank.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selected orbital indices must be unique.")
    R = Phi[:, indices]
    norm = np.sqrt(np.sum(R * R, axis=1, keepdims=True) + float(epsilon))
    return R / norm


def orbital_occupations(A: np.ndarray) -> np.ndarray:
    """
    PDF Section 1.11:
      o_k(x) = a_k(x)^2.
    """
    A = np.asarray(A, dtype=float)
    return A * A
