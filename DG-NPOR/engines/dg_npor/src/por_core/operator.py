"""
PDF Sections 1.9 and 1.16:
Discrete Predictive Orbital Problem.

Generalized eigenproblem:
  (alpha L + beta D V) phi = epsilon D phi

With u = D^(1/2) phi:
  H_N u = epsilon u

  H_N = alpha D^(-1/2) L D^(-1/2) + beta V
      = alpha L_sym + beta V

Only lambda = beta/alpha is structurally identifiable.
Default alpha=1, beta=lambda_pi.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

EPS = 1e-12


@dataclass
class PredictiveOrbitalEigensystem:
    eigenvalues: np.ndarray
    u: np.ndarray
    phi: np.ndarray
    H_N: sparse.csr_matrix
    L_sym: sparse.csr_matrix
    degree: np.ndarray


def normalized_graph_laplacian(
    W: sparse.spmatrix,
    degree: np.ndarray,
) -> sparse.csr_matrix:
    """
    L_sym = D^(-1/2) (D-W) D^(-1/2)
          = I - D^(-1/2) W D^(-1/2)
    """
    d = np.asarray(degree, dtype=float)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, EPS))
    Dm = sparse.diags(inv_sqrt)

    return (
        sparse.eye(W.shape[0], format="csr")
        - Dm @ sparse.csr_matrix(W) @ Dm
    ).tocsr()


def build_por_operator(
    W: sparse.spmatrix,
    degree: np.ndarray,
    V: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Exact PDF finite-sample operator:
      H_N = alpha L_sym + beta V.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if beta < 0:
        raise ValueError("beta must be >= 0.")

    L_sym = normalized_graph_laplacian(W, degree)
    V_op = sparse.diags(np.asarray(V, dtype=float))

    H_N = (
        float(alpha) * L_sym
        + float(beta) * V_op
    ).tocsr()

    return H_N, L_sym


def solve_por_eigensystem(
    W: sparse.spmatrix,
    degree: np.ndarray,
    V: np.ndarray,
    n_components: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: int = 42,
) -> PredictiveOrbitalEigensystem:
    """
    Solve H_N u_k = epsilon_k u_k, then recover the actual
    graph eigenfunction values

      phi_k = D^(-1/2) u_k.

    This recovery is essential because the PDF inductive extension
    in Section 1.10 is written in terms of phi_k(x_j), not u_k.
    """
    n = len(degree)
    k = min(max(1, int(n_components)), n - 1)

    H_N, L_sym = build_por_operator(
        W, degree, V, alpha=alpha, beta=beta
    )

    rng = np.random.RandomState(random_state)
    vals, vecs = eigsh(
        H_N,
        k=k,
        which="SA",
        v0=rng.normal(size=n),
    )
    order = np.argsort(vals)
    vals = np.asarray(vals[order], dtype=float)
    u = np.asarray(vecs[:, order], dtype=float)

    inv_sqrt_d = (
        1.0 / np.sqrt(np.maximum(np.asarray(degree), EPS))
    )
    phi = inv_sqrt_d[:, None] * u

    return PredictiveOrbitalEigensystem(
        eigenvalues=vals,
        u=u,
        phi=phi,
        H_N=H_N,
        L_sym=L_sym,
        degree=np.asarray(degree, dtype=float),
    )
