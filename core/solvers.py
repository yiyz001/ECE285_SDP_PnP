from __future__ import annotations
import warnings
import cvxpy.error
import numpy as np
import scipy
import casadi as cs
import cvxpy as cp
from core.manifold_utils import proper_svd
from core.model_utils import eval_h, eval_H, rot_init_multi, generate_constraints

warnings.filterwarnings('ignore')


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
            R_raw = r_prev.reshape(3, 3)
            U, _, Vt = np.linalg.svd(R_raw)
            return U @ Vt

    R_raw = r_prev.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    return U @ Vt


def sequential_qp(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500) -> np.ndarray:
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
            R_raw = r_prev.reshape(3, 3)
            U, _, Vt = np.linalg.svd(R_raw)
            return U @ Vt

        # Infeasible detection
        if not solver.stats()['success']:
            min_idx = np.argmin(np.array(costs))
            U, _, Vt = np.linalg.svd(sols[min_idx])
            return U @ Vt
        else:
            sols.append(r_prev.reshape(3, 3))
            costs.append(r_prev.T @ Omega @ r_prev)

    R_raw = r_prev.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    return U @ Vt


def original_qcqp(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500):
    """
    Reformulate the PnP problem into a QCQP and solve it directly using scipy

    :param Omega: data matrix Omega
    :param R_init: initial guess
    :param max_steps: maximum steps allowed, added to keep interface consistent

    :return: rotation matrix R
    """
    # Create objective and constraints
    # objective function, avoid lambda
    def obj(x):
        return x.T @ Omega @ x

    # Constraint function for each qc, avoid lambda
    def constraint_func(x, V, v, c):
        return x.T @ V @ x + v.T @ x + c

    # Define constraints
    V_list, v_list, c_list = generate_constraints()
    constraints = ({'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[0], v_list[0], c_list[0])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[1], v_list[1], c_list[1])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[2], v_list[2], c_list[2])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[3], v_list[3], c_list[3])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[4], v_list[4], c_list[4])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[5], v_list[5], c_list[5])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[6], v_list[6], c_list[6])},
                   {'type': 'eq', 'fun': lambda x:  constraint_func(x, V_list[7], v_list[7], c_list[7])})

    # Solve the problem
    res = scipy.optimize.minimize(obj, R_init.flatten(), method='trust-constr', constraints=constraints)
    r = res.x

    R_raw = r.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    return U @ Vt


def SDP_relaxation(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500):
    """
    SDP relaxation of the PnP-QCQP problem

    :param Omega: data matrix Omega
    :param R_init: initial guess, added to keep interface consistent
    :param max_steps: maximum steps allowed, added to keep interface consistent

    :return: rotation matrix R
    """
    # Create objective and constraints
    X = cp.Variable((9, 9), symmetric=True)
    Y = cp.Variable((10, 10), symmetric=True)
    x = cp.Variable(9)
    # objective function, avoid lambda
    obj = cp.Minimize(cp.trace(Omega @ X))
    # Create constraints
    V, v, c = generate_constraints()
    constraints = []
    for i in range(8):
        constraints.append(cp.trace(V[i] @ X) + v[i].T @ x + c[i] == 0)
    constraints.append(X >> 0)
    constraints.append(Y[0, 0] == 1)
    constraints.append(Y[1:, 1:] == X)
    constraints.append(Y[1:, 0] == x)
    constraints.append(Y[0, 1:] == x)
    constraints.append(Y >> 0)

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except cvxpy.error.SolverError:
        return np.eye(3)

    # Extract rotation matrix
    r = x.value

    R_raw = r.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    return U @ Vt


def dual_QCQP(Omega: np.ndarray, R_init: np.ndarray, max_steps: int | None = 500):
    """
    Dual of the PnP-QCQP problem

    :param Omega: data matrix Omega
    :param R_init: initial guess, added to keep interface consistent
    :param max_steps: maximum steps allowed, added to keep interface consistent

    :return: rotation matrix R
    """
    # Create objective and constraints
    gamma = cp.Variable(1)
    lamb = cp.Variable(8)
    Y = cp.Variable((10, 10), symmetric=True)
    # objective function, avoid lambda
    obj = cp.Minimize(-gamma)
    # Create constraints
    V, v, c = generate_constraints()
    y1 = -gamma
    y2 = np.zeros(9)
    y3 = Omega
    for i in range(8):
        y1 = y1 + lamb[i] * c[i]
        y2 = y2 + 0.5 * lamb[i] * v[i]
        y3 = y3 + lamb[i] * V[i]
    constraints = [Y[0, 0] == y1, Y[0, 1:] == y2, Y[1:, 0] == y2, Y[1:, 1:] == y3, Y >> 0]

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except cvxpy.error.SolverError:
        return np.array([np.inf])

    return gamma.value


def multi_search_wrapper(Omega: np.ndarray, solver: callable, max_steps: int | None = 500) -> np.ndarray:
    """
    Solve the PnP problem using multiple initializations

    :param Omega: Positive semi-definite problem data
    :param solver: Available solvers: 'riemannian_gradient_descent', 'projected_gradient_descent', 'sequential_qp'
    :param max_steps: Maximum steps for solver

    :return: Rotations matrix R
    """
    # Find optimal solution on 8-sphere
    Sigma, Lambda = np.linalg.eig(Omega)

    # Eigenvalues in ascending order
    sorted_index = np.argsort(np.abs(Sigma))

    # Generate initializations
    r_init_all = rot_init_multi(Lambda)

    # # Select acceptable initializations
    # eig_upper_bound = np.min(np.sum((r_init_all @ Omega) * r_init_all, axis=1)) / 3.0
    # init_len = np.sum(np.abs(Sigma) <= eig_upper_bound)
    # min_eig_index = sorted_index[:init_len]
    # min_eig_vec = Lambda[:, min_eig_index]

    # Initialization
    rot_ini_vec = r_init_all
    total_tasks = r_init_all.shape[0]
    costs = []
    sols = []

    for i in range(total_tasks):
        R = solver(Omega, rot_ini_vec[i].reshape(3, 3), max_steps)
        cost = (R.flatten()).T @ Omega @ (R.flatten())
        sols.append(R)
        costs.append(cost)

    # Select the best among all solutions
    best_idx = np.argmin(np.array(costs))

    return sols[best_idx]
