import numpy as np


def eval_h(x: np.ndarray) -> np.ndarray:
    """
    Implementation of the special orthogonal constrain h(x) function

    :param x: (9, ) array

    :return: (6, ) array
    """
    h = np.array([x[0:3].T @ x[0:3] - 1,
                  x[3:6].T @ x[3:6] - 1,
                  x[0:3].T @ x[3:6],
                  x[0:3].T @ x[6:9],
                  x[3:6].T @ x[6:9],
                  np.linalg.det(x.reshape(3, 3)) - 1])

    return h


def eval_H(x: np.ndarray) -> np.ndarray:
    """
    Implementation of the Jacobi of the special orthogonal constrain h(x) function

    :param x: (9, ) array

    :return: (6, 9) matrix
    """
    H = np.array([[x[0], x[1], x[2], 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, x[3], x[4], x[5], 0, 0, 0],
                  [x[3], x[4], x[5], x[0], x[1], x[2], 0, 0, 0],
                  [x[6], x[7], x[8], 0, 0, 0, x[0], x[1], x[2]],
                  [0, 0, 0, x[6], x[7], x[8], x[3], x[4], x[5]],
                  [x[4] * x[8] - x[5] * x[7], x[5] * x[6] - x[3] * x[8], x[3] * x[7] - x[4] * x[6],
                   x[2] * x[7] - x[1] * x[8], x[0] * x[8] - x[2] * x[6], x[1] * x[6] - x[0] * x[7],
                   x[1] * x[5] - x[2] * x[4], x[2] * x[3] - x[0] * x[5], x[0] * x[4] - x[1] * x[3]]
                  ])

    return H


def rot_init_multi(min_eig_vec: np.ndarray):
    """
    Generate multiple initializations based on SQPNP paper

    :param min_eig_vec: eigenvector of Omega matrix

    :return: rotation initial guesses
    """
    # Number of Eigen vectors
    num_inits = int(min_eig_vec.size / 9)

    # Solve Nearest Orthogonal Matrix Approximation Problem
    U, _, V_hat = np.linalg.svd(np.sqrt(3) * min_eig_vec.T.reshape(-1, 3, 3))
    mU, _, mV_hat = np.linalg.svd(-np.sqrt(3) * min_eig_vec.T.reshape(-1, 3, 3))
    C = np.kron(np.ones(num_inits)[:, None, None], np.eye(3))
    mC = C.copy()
    C[:, 2, 2] = np.sign(np.real(np.linalg.det(U @ V_hat)))
    mC[:, 2, 2] = np.sign(np.real(np.linalg.det(mU @ mV_hat)))
    R_ini = np.real(U @ C @ V_hat)
    mR_ini = np.real(mU @ mC @ mV_hat)
    x_ini = np.concatenate((R_ini.reshape(num_inits, 9), mR_ini.reshape(num_inits, 9)))

    return x_ini


def generate_constraints():
    """
    Cast SO(3) constraints into QC form

    :return: list V, list v, list c
    """
    # Initialize container
    V = []

    # Fill in v and c
    vE = np.zeros((9, 9))
    vE[6:, 6:] = -np.eye(3)
    v = [np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9), vE[:, 6], vE[:, 7], vE[:, 8]]
    c = [-1, -1, 0, 0, 0, 0, 0, 0]

    # Create template matrix for V_i
    for i in range(8):
        V.append(np.zeros((9, 9)))

    # Fill in V
    V[0][:3, :3] = np.eye(3)
    V[1][3:6, 3:6] = np.eye(3)
    V[2][:3, 3:6] = np.eye(3)
    V[3][:3, 6:] = np.eye(3)
    V[4][3:6, 6:] = np.eye(3)
    V[5][1, 5] = 1
    V[5][2, 4] = -1
    V[6][2, 3] = 1
    V[6][0, 5] = -1
    V[7][0, 4] = 1
    V[7][1, 3] = -1

    # Make V symmetric
    for i in range(8):
        V[i] = 0.5 * (V[i] + V[i].T)

    return V, v, c


def eval_cost(Omega: np.ndarray, R_sol: np.ndarray) -> np.ndarray:
    """
    Evaluate the cost function

    :param Omega: data matrix
    :param R_sol: solution

    :return: cost
    """

    return R_sol.flatten() @ Omega @ R_sol.flatten()


def gt_error(R_gt: np.ndarray, R_sol: np.ndarray) -> np.ndarray:
    """
    Measure the distance measured in Frobenius norm between ground truth rotation and estimated rotation

    :param R_gt: Ground truth rotation
    :param R_sol: Estimated rotation

    :return: error
    """

    return np.linalg.norm(R_gt - R_sol)


def re_projection_error(R: np.ndarray, p: np.ndarray, m: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Measure the re-projection error using the estimated pose

    :param R: rotation
    :param p: translation
    :param m: world-frame points
    :param z: observations, homogeneous

    :return: re-projection error
    """
    z_proj = np.sum(R[None, :, :] * m[:, None, :], axis=-1) + p[None, :]
    z_homo = z_proj / z_proj[:, 2][:, None]

    return np.sum(np.linalg.norm(z_homo[:, :2] - z[:, :2], axis=0))
