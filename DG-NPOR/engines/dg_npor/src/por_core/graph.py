"""
PDF Section 1.8:
Sparse self-tuning kNN graph.

w_ij = exp(-||z_i-z_j||^2 / (2 sigma_i sigma_j))
for connected pairs, zero otherwise, followed by
W <- (W + W^T)/2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


@dataclass
class SelfTuningGraph:
    W: sparse.csr_matrix
    degree: np.ndarray
    local_scale: np.ndarray
    X: np.ndarray
    nn: NearestNeighbors
    n_neighbors: int
    scale_neighbor: int


def build_self_tuning_graph(
    X: np.ndarray,
    n_neighbors: int = 20,
    scale_neighbor: Optional[int] = None,
    metric: str = "euclidean",
) -> SelfTuningGraph:
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 3:
        raise ValueError("At least 3 observations are required.")

    k = min(max(2, int(n_neighbors)), n - 1)
    k_sigma = k if scale_neighbor is None else min(
        max(1, int(scale_neighbor)), k
    )

    # +1 because each training point is its own nearest neighbour.
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="auto",
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    sigma = distances[:, k_sigma - 1].copy()
    positive = sigma[sigma > EPS]
    replacement = (
        float(np.median(positive)) if positive.size else 1.0
    )
    sigma[sigma <= EPS] = replacement

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    dij = distances.reshape(-1)

    denom = 2.0 * sigma[rows] * sigma[cols]
    values = np.exp(
        -(dij * dij) / np.maximum(denom, EPS)
    )

    # Build once in COO, then CSR. No structural CSR mutation.
    W_directed = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(n, n),
    ).tocsr()

    W = (0.5 * (W_directed + W_directed.T)).tocsr()
    W.eliminate_zeros()

    degree = np.asarray(W.sum(axis=1)).ravel()
    if np.any(degree <= EPS):
        raise RuntimeError(
            "Graph contains isolated nodes; increase n_neighbors."
        )

    return SelfTuningGraph(
        W=W,
        degree=degree,
        local_scale=sigma,
        X=X,
        nn=nn,
        n_neighbors=k,
        scale_neighbor=k_sigma,
    )


def query_affinities(
    graph: SelfTuningGraph,
    X_query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PDF Section 1.10:
      w_j(x) = K(z, z_j), d(x)=sum_j w_j(x).

    Uses the same self-tuning kernel.  sigma(x) is the distance
    to the scale-neighbour among the training neighbours.
    """
    Xq = np.asarray(X_query, dtype=float)
    k = graph.n_neighbors

    # query is not in training set, so no +1 needed.
    nnq = NearestNeighbors(
        n_neighbors=k,
        metric=graph.nn.metric,
        algorithm="auto",
    )
    nnq.fit(graph.X)
    distances, indices = nnq.kneighbors(Xq)

    k_sigma = min(graph.scale_neighbor, k)
    sigma_q = distances[:, k_sigma - 1].copy()
    positive = sigma_q[sigma_q > EPS]
    replacement = (
        float(np.median(positive)) if positive.size else 1.0
    )
    sigma_q[sigma_q <= EPS] = replacement

    A = np.zeros((len(Xq), len(graph.X)), dtype=float)

    for i in range(len(Xq)):
        idx = indices[i]
        denom = (
            2.0
            * sigma_q[i]
            * graph.local_scale[idx]
        )
        w = np.exp(
            -(distances[i] ** 2)
            / np.maximum(denom, EPS)
        )
        A[i, idx] = w

    degree_q = A.sum(axis=1)
    return A, degree_q


def query_affinities_sparse(
    graph: SelfTuningGraph,
    X_query: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Memory-safe query affinities with the same kernel as the graph.

    The v2 implementation materialized an ``n_query x n_train`` dense array
    even though every query has only ``k`` non-zero neighbours.  This version
    keeps the inductive step sparse, which matters for Higgs-sized samples.
    """
    Xq = np.asarray(X_query, dtype=float)
    k = graph.n_neighbors
    distances, indices = graph.nn.kneighbors(Xq, n_neighbors=k)

    k_sigma = min(graph.scale_neighbor, k)
    sigma_q = distances[:, k_sigma - 1].copy()
    positive = sigma_q[sigma_q > EPS]
    replacement = float(np.median(positive)) if positive.size else 1.0
    sigma_q[sigma_q <= EPS] = replacement

    rows = np.repeat(np.arange(len(Xq)), k)
    cols = indices.reshape(-1)
    denom = 2.0 * np.repeat(sigma_q, k) * graph.local_scale[cols]
    values = np.exp(
        -(distances.reshape(-1) ** 2) / np.maximum(denom, EPS)
    )

    A = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(Xq), len(graph.X)),
    ).tocsr()
    degree_q = np.asarray(A.sum(axis=1)).ravel()
    return A, degree_q
