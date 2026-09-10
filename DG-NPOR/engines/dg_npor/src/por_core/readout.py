"""
PDF Section 1.12:
Structured Orbital Measurement.

E_c = B_c^T B_c + (rho/C) I
S   = sum_c E_c
M_c = S^(-1/2) E_c S^(-1/2)

P(Y=c|x) = a_K(x)^T M_c a_K(x)

The B_c parameters are trained by NLL + optional Frobenius regularization.

This implementation uses scipy.optimize on a low-rank B_c factor.  The PDF
allows B_c in R^{r x K}; r is therefore a model parameter rather than a change
to the stated method.
"""

import numpy as np
from typing import Optional
from scipy.optimize import minimize

EPS = 1e-12


def _sym_inverse_sqrt(S: np.ndarray, floor: float = 1e-9):
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    vals = np.maximum(vals, floor)
    return (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T


def _inverse_sqrt_cache(S: np.ndarray, floor: float = 1e-9):
    """Inverse square root and its spectral Frechet derivative kernel."""
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    if np.min(vals) <= floor:
        raise RuntimeError(
            "Structured measurement normalization is numerically singular."
        )
    function_values = 1.0 / np.sqrt(vals)
    difference = vals[:, None] - vals[None, :]
    function_difference = function_values[:, None] - function_values[None, :]
    kernel = np.empty_like(difference)
    separated = np.abs(difference) > 1e-10 * np.maximum(
        1.0, np.maximum(np.abs(vals[:, None]), np.abs(vals[None, :]))
    )
    kernel[separated] = function_difference[separated] / difference[separated]
    # The divided difference tends to f'(lambda) for repeated eigenvalues.
    mean_vals = 0.5 * (vals[:, None] + vals[None, :])
    kernel[~separated] = -0.5 * mean_vals[~separated] ** (-1.5)
    inverse_sqrt = (vecs * function_values[None, :]) @ vecs.T
    return inverse_sqrt, vecs, kernel


class StructuredOrbitalMeasurement:
    def __init__(
        self,
        rank: Optional[int] = None,
        rho: float = 1e-3,
        l2: float = 1e-4,
        max_iter: int = 1200,
        n_restarts: int = 2,
        random_state: int = 42,
    ):
        self.rank = rank
        self.rho = float(rho)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.n_restarts = int(n_restarts)
        self.random_state = int(random_state)

    def __getstate__(self):
        """Exclude the fit-only local gradient closure from saved models."""
        state = self.__dict__.copy()
        state.pop("_objective_and_gradient_", None)
        return state

    def _unpack(self, theta, C, r, K):
        return theta.reshape(C, r, K)

    def _measurement_operators(self, B):
        C, r, K = B.shape
        I = np.eye(K)

        E = np.empty((C, K, K), dtype=float)
        for c in range(C):
            E[c] = B[c].T @ B[c] + (self.rho / C) * I

        S = E.sum(axis=0)
        Sinv2 = _sym_inverse_sqrt(S)

        M = np.empty_like(E)
        for c in range(C):
            M[c] = Sinv2 @ E[c] @ Sinv2

        return M

    def _probabilities(self, A, M):
        # p_ic = a_i^T M_c a_i
        P = np.column_stack([
            np.einsum("ni,ij,nj->n", A, Mc, A)
            for Mc in M
        ])
        P = np.maximum(P, EPS)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def fit(self, A, y, sample_weight=None):
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=int)

        if A.ndim != 2 or len(A) != len(y) or A.shape[1] < 2:
            raise ValueError(
                "Structured measurement needs matching observations and K >= 2."
            )
        if not np.all(np.isfinite(A)):
            raise ValueError("Structured measurement state contains non-finite values.")

        if sample_weight is None:
            weight = np.ones(len(y), dtype=float)
        else:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.shape != (len(y),):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(~np.isfinite(weight)) or np.any(weight < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
        if weight.sum() <= EPS:
            raise ValueError("sample_weight must have positive total weight.")
        weight = weight / weight.mean()

        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError(
                "StructuredOrbitalMeasurement expects encoded labels 0..C-1."
            )

        C = len(self.classes_)
        K = A.shape[1]
        r = (
            min(K, 2)
            if self.rank is None
            else min(max(1, int(self.rank)), K)
        )
        self.rank_ = r

        rng = np.random.RandomState(self.random_state)

        def objective_and_gradient(theta):
            B = self._unpack(theta, C, r, K)
            identity = np.eye(K)
            E = np.asarray([
                B[c].T @ B[c] + (self.rho / C) * identity
                for c in range(C)
            ])
            S = E.sum(axis=0)
            Sinv2, eigenvectors, frechet_kernel = _inverse_sqrt_cache(S)
            M = np.asarray([Sinv2 @ E[c] @ Sinv2 for c in range(C)])
            P = self._probabilities(A, M)

            point_nll = -np.log(
                np.maximum(P[np.arange(len(y)), y], EPS)
            )
            nll = float(np.sum(weight * point_nll) / np.sum(weight))
            reg = self.l2 * float(np.sum(B * B))
            objective = nll + reg

            # Reverse derivative of L through M_c=T E_c T and T=S^(-1/2).
            # This avoids scipy's finite-difference cost, which scales as
            # O(C*r*K) full objective evaluations per L-BFGS iteration.
            gradient_M = np.zeros_like(M)
            normalizer = float(np.sum(weight))
            for c in range(C):
                mask = y == c
                coefficients = np.zeros(len(y), dtype=float)
                coefficients[mask] = (
                    -weight[mask]
                    / np.maximum(P[mask, c], EPS)
                    / normalizer
                )
                gradient_M[c] = A.T @ (coefficients[:, None] * A)
                gradient_M[c] = 0.5 * (
                    gradient_M[c] + gradient_M[c].T
                )

            gradient_T = np.zeros((K, K), dtype=float)
            direct_gradient_E = np.empty_like(E)
            for c in range(C):
                direct_gradient_E[c] = Sinv2 @ gradient_M[c] @ Sinv2
                gradient_T += (
                    gradient_M[c] @ Sinv2 @ E[c]
                    + E[c] @ Sinv2 @ gradient_M[c]
                )
            gradient_T = 0.5 * (gradient_T + gradient_T.T)
            rotated = eigenvectors.T @ gradient_T @ eigenvectors
            gradient_S = eigenvectors @ (
                frechet_kernel * rotated
            ) @ eigenvectors.T
            gradient_S = 0.5 * (gradient_S + gradient_S.T)
            gradient_B = np.empty_like(B)
            for c in range(C):
                gradient_E = direct_gradient_E[c] + gradient_S
                gradient_E = 0.5 * (gradient_E + gradient_E.T)
                gradient_B[c] = 2.0 * B[c] @ gradient_E + 2.0 * self.l2 * B[c]
            return float(objective), gradient_B.reshape(-1)

        # Retained for deterministic finite-difference audit tests.
        self._objective_and_gradient_ = objective_and_gradient
        best = None
        for restart in range(max(1, self.n_restarts)):
            theta0 = 0.1 * rng.normal(size=C * r * K)

            # Mild class-oriented initialization for first row.
            for c in range(C):
                mu = A[y == c].mean(axis=0)
                nrm = np.linalg.norm(mu)
                if nrm > EPS:
                    start = c * r * K
                    theta0[start:start + K] += mu / nrm

            res = minimize(
                objective_and_gradient,
                theta0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter,
                    "maxfun": max(1000, 5 * self.max_iter),
                    "ftol": 1e-8,
                    "gtol": 1e-6,
                    "maxls": 20,
                },
            )

            if best is None or res.fun < best.fun:
                best = res

        if best is None or not np.isfinite(best.fun):
            raise RuntimeError("Structured measurement optimization failed.")
        self.optimization_result_ = best
        self.B_ = self._unpack(best.x, C, r, K)
        self.M_ = self._measurement_operators(self.B_)
        self.train_objective_ = float(best.fun)
        self.gradient_norm_ = float(np.linalg.norm(best.jac))
        self.n_iterations_ = int(best.nit)
        self.n_function_evaluations_ = int(best.nfev)
        identity_error = np.max(
            np.abs(self.M_.sum(axis=0) - np.eye(K))
        )
        minimum_eigenvalue = min(
            float(np.min(np.linalg.eigvalsh(operator)))
            for operator in self.M_
        )
        if identity_error > 1e-6 or minimum_eigenvalue < -1e-8:
            raise RuntimeError("Structured measurement constraints are violated.")
        return self

    def predict_proba(self, A):
        return self._probabilities(
            np.asarray(A, dtype=float),
            self.M_,
        )

    def predict(self, A):
        P = self.predict_proba(A)
        return np.argmax(P, axis=1)
