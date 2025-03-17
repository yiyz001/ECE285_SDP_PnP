from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.data_utils import generate_data, construct_data_matrix
from core.model_utils import eval_h
from core.manifold_utils import proper_svd


mpl.use("Qt5Agg")


def projected_gradient_descent(Omega: np.ndarray, R_init: np.ndarray,
                               max_steps: int | None = 500, step_size: float | None = 0.1) -> list:
    """
    Solve the PnP problem using Riemannian gradient descent.

    :param Omega: Data matrix Omega
    :param R_init: Initial guess of rotation matrix
    :param max_steps: Maximum steps allowed for projected gradient descent
    :param step_size: Step size for projected gradient descent

    :return: rotation matrix R
    """
    # Initialize solution container
    sols = [R_init]

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
        sols.append(r_prev.reshape(3, 3))

        # Early termination
        if np.linalg.norm(grad_euclidean) <= 1e-3:
            return sols

    return sols


if __name__ == '__main__':
    np.random.seed(10)
    # Generate synthesis data
    gt_list, obsv_z, lmark_m = generate_data(num_obsv=10)

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])

    # Define and solve the PnP problem
    R_sol_rgd = projected_gradient_descent(Omega=list_Omega[0], R_init=np.eye(3), max_steps=500)

    # Evaluate constraints
    h_eval = []
    for R in R_sol_rgd:
        h_eval.append(eval_h(R.flatten()))

    plt.ion()
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    fig.suptitle("PGD Constraint Satisfaction")
    ax1 = fig.add_subplot(111, frameon=True)
    ax1.plot(np.arange(0, len(R_sol_rgd)), np.array(h_eval), label='PGD')

    plt.show(block=True)
