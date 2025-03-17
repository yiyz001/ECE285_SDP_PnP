from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.data_utils import generate_data, construct_data_matrix
from core.model_utils import eval_h, eval_H
import casadi as cs


mpl.use("Qt5Agg")


def sequential_qp(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500) -> list:
    """
    Solve the PnP problem using Riemannian gradient descent.

    :param Omega: Data matrix Omega
    :param R_init: Initial guess of rotation matrix
    :param max_steps: Maximum steps allowed

    :return: rotation matrix R
    """
    # Init data container
    costs = []
    sols = []

    # Initialize Casadi
    H = cs.DM.ones(9, 9)
    A = cs.DM.ones(6, 9)
    opts = {'osqp.verbose': False, 'print_time': False, 'error_on_fail': False}
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    solver = cs.conic('S', 'osqp', qp, opts)

    # Initialize initial guess of R
    r_prev = R_init.flatten()
    sols.append(R_init)
    costs.append(r_prev.T @ Omega @ r_prev)

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
            sols.append(r_prev.reshape(3, 3))
            return sols

        # Infeasible detection
        if not solver.stats()['success']:
            return sols
        else:
            sols.append(r_prev.reshape(3, 3))
            costs.append(r_prev.T @ Omega @ r_prev)

    return sols


if __name__ == '__main__':
    np.random.seed(10)
    # Generate synthesis data
    gt_list, obsv_z, lmark_m = generate_data(num_obsv=10)

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])

    # Define and solve the PnP problem
    R_sol_rgd = sequential_qp(Omega=list_Omega[0], R_init=np.eye(3), max_steps=500)

    # Evaluate constraints
    h_eval = []
    for R in R_sol_rgd:
        h_eval.append(eval_h(R.flatten()))

    plt.ion()
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    fig.suptitle("SQP Constraint Satisfaction")
    ax1 = fig.add_subplot(111, frameon=True)
    ax1.plot(np.arange(0, len(R_sol_rgd)), np.array(h_eval), label='PGD')

    plt.show(block=True)
