from __future__ import annotations
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.data_utils import generate_data, construct_data_matrix
from core.model_utils import eval_h


mpl.use("Qt5Agg")


def riemannian_gradient_descent(Omega: np.ndarray, R_init: np.ndarray,
                                max_steps: int | None = 500, step_size: float | None = 0.1) -> list:
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
            return sols

    return sols


if __name__ == '__main__':
    np.random.seed(10)
    # Generate synthesis data
    gt_list, obsv_z, lmark_m = generate_data(num_obsv=10)

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])

    # Define and solve the PnP problem
    R_sol_rgd = riemannian_gradient_descent(Omega=list_Omega[0], R_init=np.eye(3), max_steps=500)

    # Evaluate constraints
    h_eval = []
    for R in R_sol_rgd:
        h_eval.append(eval_h(R.flatten()))

    plt.ion()
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    fig.suptitle("RGD Constraint Satisfaction")
    ax1 = fig.add_subplot(111, frameon=True)
    ax1.plot(np.arange(0, len(R_sol_rgd)), np.array(h_eval), label='PGD')

    plt.show(block=True)
