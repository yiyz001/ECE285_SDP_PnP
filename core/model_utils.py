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
