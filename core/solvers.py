from __future__ import annotations
import numpy as np
import scipy
import casadi as cs
from core.manifold_utils import proper_svd
from core.model_utils import eval_h, eval_H


def riemannian_gradient_descent(Omega: np.ndarray, R_init: np.ndarray,
                                max_steps: int | None = 500, step_size: float | None = 0.1) -> np.ndarray:
    """
    Solve the PnP problem using Riemannian gradient descent.

    :param Omega: Data matrix Omega
    :param R_init: Initial guess of rotation matrix
    :param max_steps: Maximum steps allowed for Riemannian gradient descent
    :param step_size: Step size for Riemannian gradient descent

    :return: rotation matrix R
    """
    # Init data container
    cost = []
    sols = []

    # Flatten rotation matrix
    r_prev = R_init.flatten()
    sols.append(R_init)
    cost.append(r_prev.T @ Omega @ r_prev)

    for i in range(max_steps):
        # Reconstruct rotation matrix
        R_prev = r_prev.reshape(3, 3)

        # Obtain the Euclidean gradient of the problem
        grad_euclidean = (2 * Omega @ r_prev).reshape(3, 3)

        # Perform orthogonal projection to tangent space at identity
        grad_riemannian = 0.5 * (R_prev.T @ grad_euclidean - grad_euclidean.T @ R_prev)

        # Calculate new R
        R_new = R_prev @ scipy.linalg.expm(step_size * grad_riemannian)

        # Perform update
        r_prev = R_new.flatten()

        # Cache results
        sols.append(r_prev.reshape(3, 3))
        cost.append(r_prev.T @ Omega @ r_prev)

        # Early termination
        if np.trace(grad_riemannian.T @ grad_riemannian) <= 1e-3:
            return r_prev.reshape(3, 3)

    return r_prev.reshape(3, 3)


def projected_gradient_descent(Omega: np.ndarray, R_init: np.ndarray,
                               max_steps: int | None = 500, step_size: float | None = 0.1) -> np.ndarray:
    """
    Solve the PnP problem using Riemannian gradient descent.

    :param Omega: Data matrix Omega
    :param R_init: Initial guess of rotation matrix
    :param max_steps: Maximum steps allowed for projected gradient descent
    :param step_size: Step size for projected gradient descent

    :return: rotation matrix R
    """
    # Flatten rotation matrix
    r_prev = R_init.flatten()

    for i in range(max_steps):
        # Obtain the Euclidean gradient of the problem
        grad_euclidean = (2 * Omega @ r_prev)

        # Obtain new r
        r_updated = r_prev - step_size * grad_euclidean

        # Project back to SO(3) using proper singular value decomposition
        U, _, Vt = proper_svd(r_updated.reshape(3, 3)[None, :, :])

        # Perform update
        r_prev = (U @ Vt).flatten()

        # Early termination
        if np.linalg.norm(grad_euclidean) <= 1e-3:
            return r_prev.reshape(3, 3)

    return r_prev.reshape(3, 3)


def sequential_qp(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500) -> np.ndarray:
    """
    Solve the PnP problem using Riemannian gradient descent.

    :param Omega: Data matrix Omega
    :param R_init: Initial guess of rotation matrix
    :param max_steps: Maximum steps allowed

    :return: rotation matrix R
    """
    # Initialize Casadi
    H = cs.DM.ones(9, 9)
    A = cs.DM.ones(6, 9)
    opts = {'osqp.verbose': False, 'print_time': False, 'error_on_fail': False}
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    solver = cs.conic('S', 'osqp', qp, opts)

    # Initialize initial guess of R
    r_prev = R_init.flatten()

    for i in range(max_steps):
        # Evaluate constraint function h
        h_r = eval_h(r_prev)

        # Solve QP
        sol = solver(h=2 * Omega, g=2 * Omega @ r_prev, a=eval_H(r_prev), lba=-h_r-1e-1, uba=-h_r+1e-1)

        # Extract solution
        delta = np.array(sol['x']).flatten()

        # Update
        r_prev = r_prev + delta

        # Early termination
        if np.linalg.norm(delta) <= 1e-3:
            return r_prev.reshape(3, 3)

    return r_prev.reshape(3, 3)
