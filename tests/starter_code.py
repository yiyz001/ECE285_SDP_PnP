from utils.data_utils import generate_data, kitti_data, construct_data_matrix, generate_noisy_data
from core.model_utils import re_projection_error
from core.solvers import riemannian_gradient_descent, projected_gradient_descent, sequential_qp, multi_search_wrapper
from core.solvers import original_qcqp, SDP_relaxation, dual_QCQP
import numpy as np


if __name__ == '__main__':
    # Generate synthesis data
    gt_list, obsv_z, lmark_m = generate_noisy_data(num_obsv=10, noise_level=0.001)

    # Load data from KITTI, we do not have ground-truth this time
    # obsv_z, lmark_m = kitti_data()

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])

    # Define and solve the PnP problem
    R_sol_sqp = sequential_qp(Omega=list_Omega[0], R_init=np.eye(3), max_steps=1000)
    R_sol = multi_search_wrapper(Omega=list_Omega[0], solver=riemannian_gradient_descent, max_steps=500)
    # R_sol = SDP_relaxation(Omega=list_Omega[0], R_init=np.eye(3))
    # gamma = dual_QCQP(Omega=list_Omega[0], R_init=np.eye(3))

    print("Difference from GT: ", np.linalg.norm(gt_list[0] - R_sol))
    # print("SDP relaxation cost: ", R_sol.flatten().T @ list_Omega[0] @ R_sol.flatten().T)
    # print("Dual prolmen report lower bound: ", gamma.squeeze())
