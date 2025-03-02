import numpy as np


def hat_map(x: np.ndarray) -> np.ndarray:
    """
    Hat map R^3 -> so(3)

    :param x: input of shape (n, 3)

    :return: skew-symmetric matrix of shape (n, 3, 3)
    """
    assert x.shape[1] == 3, "Dimension mismatch"

    X = np.zeros(x.shape + (3,))
    X[..., 0, 1] = -x[:, 2]
    X[..., 0, 2] = x[:, 1]
    X[..., 1, 0] = x[:, 2]
    X[..., 1, 2] = -x[:, 0]
    X[..., 2, 0] = -x[:, 1]
    X[..., 2, 1] = x[:, 0]

    return X
