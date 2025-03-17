import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from core.manifold_utils import hat_map, proper_svd


mpl.use("Qt5Agg")


def wrap_angle_pmp(angle_vec):
    """
    npla.normalize angle in radian to [-pi, pi)
    angle_vec: angle description in radian
    """
    index = angle_vec <= -0.01
    angle_vec[index] = angle_vec[index] + 2 * np.pi
    return angle_vec


if __name__ == "__main__":
    # Start from this rotation matrix
    R_init = np.eye(3)

    # Axis angle rotation
    omega = np.array([0, 0, 1])

    # Time steps
    t_steps = np.linspace(0, 2 * np.pi, 501)

    # Start evaluation
    rgd_sols = [R_init]
    pgd_sols = [R_init]
    rgd_sols_rvec = [np.zeros(3)]
    pgd_sols_rvec = [np.zeros(3)]
    for i in range(500):
        # RGD update
        R_rgd_new = R_init @ scipy.linalg.expm(t_steps[i] * hat_map(omega[None, :]).squeeze())
        rgd_sols.append(R_rgd_new)
        R_rgd_scipy = scipy.spatial.transform.Rotation.from_matrix(R_rgd_new)
        rgd_sols_rvec.append(R_rgd_scipy.as_rotvec())

        # PGD update
        un_R_pgd_new = R_init + t_steps[i] * R_init @ hat_map(omega[None, :]).squeeze()
        U, _, Vt = proper_svd(un_R_pgd_new)
        R_pgd_new = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ Vt)]]) @ Vt
        pgd_sols.append(R_pgd_new.squeeze())
        R_pgd_scipy = scipy.spatial.transform.Rotation.from_matrix(R_pgd_new)
        pgd_sols_rvec.append(R_pgd_scipy.as_rotvec())

    plt.ion()
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    fig.suptitle("PGD vs RGD")
    ax1 = fig.add_subplot(111, frameon=True)
    ax1.plot(t_steps, np.array(rgd_sols_rvec)[:, 2], label='RGD')
    ax1.plot(t_steps, np.array(pgd_sols_rvec)[:, 2], label='PGD')
    ax1.legend(loc="best", fontsize=13)

    plt.show(block=True)
