from utils.data_utils import generate_noisy_data, kitti_data, construct_data_matrix
from core.solvers import riemannian_gradient_descent, projected_gradient_descent, sequential_qp, multi_search_wrapper
from core.solvers import original_qcqp, SDP_relaxation, dual_QCQP
import numpy as np
from tqdm import tqdm
import pickle
import time
import sys


if __name__ == '__main__':
    # Prepare log file
    f = open("results/out_low_noise.txt", "w")
    sys.stdout = f

    # Test settings
    test_number = 50
    num_observation = [3, 4, 5, 8, 10]
    test_results = {"prob_data": {}, "pgd": {}, "pgd_multi": {}, "rgd": {}, "rgd_multi": {}, "sqp": {}, "sqp_multi": {},
                    "qcqp": {}, "qcqp_multi": {}, "sdp": {}, "dual": {}}

    for num_obsv in num_observation:
        print("-------- Test With %i Observations --------\n" % num_obsv)
        pbar = tqdm(total=test_number)

        # Result container
        problem_data = {"Omega": [], "P": [], "observations": [], "landmarks": [], "gt_R": [], "gt_p": [], "gt_z": []}
        result_pgd = []
        result_pgd_multi = []
        result_rgd = []
        result_rgd_multi = []
        result_sqp = []
        result_sqp_multi = []
        result_qcqp = []
        result_qcqp_multi = []
        result_sdp = []
        result_dual = []

        # Time
        time_pgd = []
        time_pgd_multi = []
        time_rgd = []
        time_rgd_multi = []
        time_sqp = []
        time_sqp_multi = []
        time_qcqp = []
        time_qcqp_multi = []
        time_sdp = []
        time_dual = []

        # Fail
        sdp_fail = 0
        dual_fail = 0

        # Start test
        for itera in range(test_number):
            # Generate synthesis data
            gt_list, obsv_z, lmark_m = generate_noisy_data(num_obsv, 0.005)

            # Get P and Omega matrices
            list_P, list_Omega = construct_data_matrix([obsv_z[:, :, None]], [lmark_m[:, :, None]])
            Sigma, Lambda = np.linalg.eig(list_Omega[0])
            U, _, V_hat = np.linalg.svd(np.sqrt(3) * Lambda[-1].reshape(-1, 3, 3))
            C = np.eye(3)
            C[2, 2] = np.sign(np.real(np.linalg.det(U @ V_hat)))
            R_ini = np.real(U @ C @ V_hat)

            # Save problem data
            problem_data["Omega"].append(list_Omega[0])
            problem_data["P"].append(list_P[0])
            problem_data["observations"].append(obsv_z)
            problem_data["landmarks"].append(lmark_m)
            problem_data["gt_R"].append(gt_list[0])
            problem_data["gt_p"].append(gt_list[1])
            problem_data["gt_z"].append(gt_list[2])

            """
            Solver the PnP problem using multiple methods
            """
            # Projected gradient descent
            ts = time.time()
            R_pgd = projected_gradient_descent(Omega=list_Omega[0], R_init=R_ini, max_steps=300, step_size=0.1)
            time_pgd.append(time.time() - ts)
            result_pgd.append(R_pgd)

            ts = time.time()
            R_pgd_multi = multi_search_wrapper(Omega=list_Omega[0], solver=projected_gradient_descent, max_steps=300)
            time_pgd_multi.append(time.time() - ts)
            result_pgd_multi.append(R_pgd_multi)

            # Riemannian gradient descent
            ts = time.time()
            R_rgd = riemannian_gradient_descent(Omega=list_Omega[0], R_init=R_ini, max_steps=300)
            time_rgd.append(time.time() - ts)
            result_rgd.append(R_rgd)

            ts = time.time()
            R_rgd_multi = multi_search_wrapper(Omega=list_Omega[0], solver=riemannian_gradient_descent, max_steps=300)
            time_rgd_multi.append(time.time() - ts)
            result_rgd_multi.append(R_rgd_multi)

            # Sequential QP
            ts = time.time()
            R_sqp = sequential_qp(Omega=list_Omega[0], R_init=R_ini, max_steps=300)
            time_sqp.append(time.time() - ts)
            result_sqp.append(R_sqp)

            ts = time.time()
            R_sqp_multi = multi_search_wrapper(Omega=list_Omega[0], solver=sequential_qp, max_steps=300)
            time_sqp_multi.append(time.time() - ts)
            result_sqp_multi.append(R_sqp_multi)

            # QCQP
            ts = time.time()
            R_qcqp = original_qcqp(Omega=list_Omega[0], R_init=R_ini, max_steps=300)
            time_qcqp.append(time.time() - ts)
            result_qcqp.append(R_qcqp)

            ts = time.time()
            R_qcqp_multi = multi_search_wrapper(Omega=list_Omega[0], solver=original_qcqp, max_steps=300)
            time_qcqp_multi.append(time.time() - ts)
            result_qcqp_multi.append(R_qcqp_multi)

            # SDP
            ts = time.time()
            R_sdp = SDP_relaxation(Omega=list_Omega[0], R_init=R_ini, max_steps=300)
            time_sdp.append(time.time() - ts)
            result_sdp.append(R_sdp)
            if np.array_equal(R_sdp, np.eye(3)):
                sdp_fail = sdp_fail + 1

            # Dual
            ts = time.time()
            c_dual = dual_QCQP(Omega=list_Omega[0], R_init=R_ini, max_steps=300)
            time_dual.append(time.time() - ts)
            result_dual.append(c_dual)
            if c_dual == np.inf:
                dual_fail = dual_fail + 1

            pbar.update(1)

        test_results["prob_data"][num_obsv] = problem_data
        test_results["pgd"][num_obsv] = result_pgd
        test_results["pgd_multi"][num_obsv] = result_pgd_multi
        test_results["rgd"][num_obsv] = result_rgd
        test_results["rgd_multi"][num_obsv] = result_rgd_multi
        test_results["sqp"][num_obsv] = result_sqp
        test_results["sqp_multi"][num_obsv] = result_sqp_multi
        test_results["qcqp"][num_obsv] = result_qcqp
        test_results["qcqp_multi"][num_obsv] = result_qcqp_multi
        test_results["sdp"][num_obsv] = result_sdp
        test_results["dual"][num_obsv] = result_dual

        print("\n----------------- Test Done -----------------\n")
        print("------------------ Summary ------------------\n")
        print("PGD Average Time: %2f \n" % (np.sum(np.array(time_pgd)) / test_number))
        print("PGD Multi Average Time: %2f \n" % (np.sum(np.array(time_pgd_multi)) / test_number))
        print("RGD Average Time: %2f \n" % (np.sum(np.array(time_rgd)) / test_number))
        print("RGD Multi Average Time: %2f \n" % (np.sum(np.array(time_rgd_multi)) / test_number))
        print("SQP Average Time: %2f \n" % (np.sum(np.array(time_sqp)) / test_number))
        print("SQP Multi Average Time: %2f \n" % (np.sum(np.array(time_sqp_multi)) / test_number))
        print("QCQP Average Time: %2f \n" % (np.sum(np.array(time_qcqp)) / test_number))
        print("QCQP Multi Average Time: %2f \n" % (np.sum(np.array(time_qcqp_multi)) / test_number))
        print("SDP Average Time: %2f \n" % (np.sum(np.array(time_sdp)) / test_number))
        print("Dual Average Time: %2f \n" % (np.sum(np.array(time_dual)) / test_number))
        print("----------------- Fail Cases -----------------\n")
        print("SDP Fail: %i Out of %i Tests \n" % (sdp_fail, test_number))
        print("Dual Fail: %i Out of %i Tests \n" % (dual_fail, test_number))
        print("-------------- End of Summary ----------------\n")

        pbar.close()

    with open('results/results_low_noise.pkl', 'wb') as handle:
        pickle.dump(test_results, handle)

    f.close()
