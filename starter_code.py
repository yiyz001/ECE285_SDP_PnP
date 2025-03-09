from utils.data_utils import generate_data, kitti_data, construct_data_matrix
from core.solvers import riemannian_gradient_descent, projected_gradient_descent, sequential_qp, multi_search_wrapper
import numpy as np


if __name__ == '__main__':
    # Generate synthesis data
    gt_list, obsv_z, lmark_m = generate_data(10)

    # Load data from KITTI, we do not have ground-truth this time
    # obsv_z, lmark_m = kitti_data()

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])

    # Define and solve the PnP problem
    # R_sol = sequential_qp(Omega=list_Omega[0], R_init=np.eye(3), max_steps=1000)
    R_sol = multi_search_wrapper(Omega=list_Omega[0], solver=riemannian_gradient_descent, max_steps=500)

    print("Difference from GT: ", np.linalg.norm(gt_list[0] - R_sol))
    print(R_sol)
