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


def proper_svd(x: np.ndarray):
    """
    Proper singular value decomposition

    :param x: input array of shape (n, dim, dim)


    :return: U of shape (n, dim, dim), Sof shape (n, dim, dim), V.T of shape (n, dim, dim)
    """
    # Singular value decomposition
    Up, Sp, Vt = np.linalg.svd(x)

    # Make proper
    U = Up.copy()
    U[..., -1] = U[..., -1] * np.sign(np.linalg.det(Up))

    S = Sp.copy()
    S[..., -1] = S[..., -1] * np.sign(np.linalg.det(Up) * np.linalg.det(Vt))

    V = np.moveaxis(Vt, -1, -2).copy()
    V[..., -1] = V[..., -1] * np.sign(np.linalg.det(Vt))

    return U, S, np.moveaxis(V, -1, -2)
