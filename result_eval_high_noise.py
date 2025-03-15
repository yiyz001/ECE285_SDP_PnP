import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from core.model_utils import eval_cost, gt_error, re_projection_error


mpl.use("Qt5Agg")


if __name__ == '__main__':
    with open('results/results_high_noise.pkl', 'rb') as handle:
        results = pickle.load(handle)

    prob_data = results['prob_data']
    pgd = results['pgd']
    pgd_multi = results['pgd_multi']
    rgd = results['rgd']
    rgd_multi = results['rgd_multi']
    sqp = results['sqp']
    sqp_multi = results['sqp_multi']
    qcqp = results['qcqp']
    qcqp_multi = results['qcqp_multi']
    sdp = results['sdp']
    dual = results['dual']

    for keys in prob_data:
        """
        Evaluation metric
        """
        # PGD
        sol_dist_pgd = []
        cost_pgd = []
        re_proj_error_pgd = []

        # PGD Multi
        sol_dist_pgd_multi = []
        cost_pgd_multi = []
        re_proj_error_pgd_multi = []

        # RGD
        sol_dist_rgd = []
        cost_rgd = []
        re_proj_error_rgd = []

        # RGD Multi
        sol_dist_rgd_multi = []
        cost_rgd_multi = []
        re_proj_error_rgd_multi = []

        # SQP
        sol_dist_sqp = []
        cost_sqp = []
        re_proj_error_sqp = []

        # SQP Multi
        sol_dist_sqp_multi = []
        cost_sqp_multi = []
        re_proj_error_sqp_multi = []

        # QCQP
        sol_dist_qcqp = []
        cost_qcqp = []
        re_proj_error_qcqp = []

        # QCQP Multi
        sol_dist_qcqp_multi = []
        cost_qcqp_multi = []
        re_proj_error_qcqp_multi = []

        # SDP
        sol_dist_sdp = []
        cost_sdp = []
        re_proj_error_sdp = []

        cost_dual = dual[keys]

        for inst_num in range(len(pgd[keys])):
            Omega = prob_data[keys]["Omega"][inst_num]
            P = prob_data[keys]["P"][inst_num]
            z = prob_data[keys]["gt_z"][inst_num]
            m = prob_data[keys]["landmarks"][inst_num]
            gt_R = prob_data[keys]["gt_R"][inst_num]
            gt_p = prob_data[keys]["gt_p"][inst_num]

            # PGD
            sol_pgd = pgd[keys][inst_num]
            sol_dist_pgd.append(gt_error(gt_R, sol_pgd))
            cost_pgd.append(eval_cost(Omega, sol_pgd))
            re_proj_error_pgd.append(re_projection_error(sol_pgd, P @ sol_pgd.flatten(), m, z))

            # PGD Multi
            sol_pgd_multi = pgd_multi[keys][inst_num]
            sol_dist_pgd_multi.append(gt_error(gt_R, sol_pgd_multi))
            cost_pgd_multi.append(eval_cost(Omega, sol_pgd_multi))
            re_proj_error_pgd_multi.append(re_projection_error(sol_pgd_multi, P @ sol_pgd_multi.flatten(), m, z))

            # RGD
            sol_rgd = rgd[keys][inst_num]
            sol_dist_rgd.append(gt_error(gt_R, sol_rgd))
            cost_rgd.append(eval_cost(Omega, sol_rgd))
            re_proj_error_rgd.append(re_projection_error(sol_rgd, P @ sol_rgd.flatten(), m, z))

            # RGD Multi
            sol_rgd_multi = rgd_multi[keys][inst_num]
            sol_dist_rgd_multi.append(gt_error(gt_R, sol_rgd_multi))
            cost_rgd_multi.append(eval_cost(Omega, sol_rgd_multi))
            re_proj_error_rgd_multi.append(re_projection_error(sol_rgd_multi, P @ sol_rgd_multi.flatten(), m, z))

            # SQP
            sol_sqp = sqp[keys][inst_num]
            sol_dist_sqp.append(gt_error(gt_R, sol_sqp))
            cost_sqp.append(eval_cost(Omega, sol_sqp))
            re_proj_error_sqp.append(re_projection_error(sol_sqp, P @ sol_sqp.flatten(), m, z))

            # SQP Multi
            sol_sqp_multi = sqp_multi[keys][inst_num]
            sol_dist_sqp_multi.append(gt_error(gt_R, sol_sqp_multi))
            cost_sqp_multi.append(eval_cost(Omega, sol_sqp_multi))
            re_proj_error_sqp_multi.append(re_projection_error(sol_sqp_multi, P @ sol_sqp_multi.flatten(), m, z))

            # QCQP
            sol_qcqp = qcqp[keys][inst_num]
            sol_dist_qcqp.append(gt_error(gt_R, sol_qcqp))
            cost_qcqp.append(eval_cost(Omega, sol_qcqp))
            re_proj_error_qcqp.append(re_projection_error(sol_qcqp, P @ sol_qcqp.flatten(), m, z))

            # QCQP Multi
            sol_qcqp_multi = qcqp_multi[keys][inst_num]
            sol_dist_qcqp_multi.append(gt_error(gt_R, sol_qcqp_multi))
            cost_qcqp_multi.append(eval_cost(Omega, sol_qcqp_multi))
            re_proj_error_qcqp_multi.append(re_projection_error(sol_qcqp_multi, P @ sol_qcqp_multi.flatten(), m, z))

            # SDP
            sol_sdp = sdp[keys][inst_num]
            sol_dist_sdp.append(gt_error(gt_R, sol_sdp))
            cost_sdp.append(eval_cost(Omega, sol_sdp))
            re_proj_error_sdp.append(re_projection_error(sol_sdp, P @ sol_sdp.flatten(), m, z))

        plt.ion()
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        fig.suptitle("Test with %i Observations" % keys)
        ax1 = fig.add_subplot(131, frameon=True)
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_pgd), label='PGD')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_pgd_multi), label='PGD_Multi')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_rgd), label='RGD')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_rgd_multi), label='RGD_Multi')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_sqp), label='SQP')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_sqp_multi), label='SQP_Multi')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_qcqp), label='QCQP')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_qcqp_multi), label='QCQP_Multi')
        ax1.plot(np.arange(0, len(pgd[keys])), np.array(sol_dist_sdp), label='SDP')
        # ax1.legend(loc="best", fontsize=13)
        ax1.set_title('Distance to Ground Truth Rotation')

        ax2 = fig.add_subplot(132, frameon=True)
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_pgd), label='PGD')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_pgd_multi), label='PGD_Multi')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_rgd), label='RGD')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_rgd_multi), label='RGD_Multi')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_sqp), label='SQP')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_sqp_multi), label='SQP_Multi')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_qcqp), label='QCQP')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_qcqp_multi), label='QCQP_Multi')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_sdp), label='SDP')
        ax2.plot(np.arange(0, len(pgd[keys])), np.array(cost_dual), label='Dual')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=15)
        # ax2.legend(loc="best", fontsize=13)
        ax2.set_title('Cost Function Value')

        ax3 = fig.add_subplot(133, frameon=True)
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_pgd), label='PGD')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_pgd_multi), label='PGD_Multi')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_rgd), label='RGD')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_rgd_multi), label='RGD_Multi')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_sqp), label='SQP')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_sqp_multi), label='SQP_Multi')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_qcqp), label='QCQP')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_qcqp_multi), label='QCQP_Multi')
        ax3.plot(np.arange(0, len(pgd[keys])), np.array(re_proj_error_sdp), label='SDP')
        # ax3.legend(loc="best", fontsize=13)
        ax3.set_title('Re-Projection Error')

        fig_hist = plt.figure(figsize=(12, 12), facecolor='white')
        fig_hist.suptitle("Cost Evaluation with %i Observations" % keys)
        ax_hist1 = fig_hist.add_subplot(221, frameon=True)
        ax_hist1.hist(np.array(cost_rgd_multi))
        ax_hist1.set_title('RGD Multi')
        ax_hist2 = fig_hist.add_subplot(222, frameon=True)
        ax_hist2.hist(np.array(cost_sqp_multi))
        ax_hist2.set_title('SQP Multi')
        ax_hist3 = fig_hist.add_subplot(223, frameon=True)
        ax_hist3.hist(np.array(cost_qcqp_multi))
        ax_hist3.set_title('QCQP Multi')
        ax_hist4 = fig_hist.add_subplot(224, frameon=True)
        ax_hist4.hist(np.array(cost_sdp))
        ax_hist4.set_title('SDP')

        fig_dual = plt.figure(figsize=(8, 8), facecolor='white')
        fig_dual.suptitle("Cost Evaluation with %i Observations" % keys)
        ax_dual = fig_dual.add_subplot(111, frameon=True)
        ax_dual.plot(np.arange(0, len(pgd[keys])), np.array(cost_sdp), label='SDP')
        ax_dual.plot(np.arange(0, len(pgd[keys])), np.array(cost_dual), label='Dual')
        ax_dual.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        plt.show(block=True)

