#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:29:20 2025

@author: martin
"""


import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from scipy.stats import ecdf

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from Benjamini_Hochberg import Storeys_correction, Benjamini_Hochberg_procedure
from ConformalContaminationTestModule import ConformalContaminationTest


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    color_list = ["tab:red", "tab:olive", "tab:purple", "tab:orange"]
    name_list = ["Storey", "Quantile", "Summation", "Fisher"]

    savename = "retina/retina_AdaDetect_LR_discrete_uniform"
    savename_list = savename.split("_")

    gamma = 0.5 # Storey's hyperparameter conformal contamination test
    num_rejection_thresholds = 40
    significance_levels = np.linspace(0, 0.975, num_rejection_thresholds)
    K = 11
    K0 = None
    use_true = False
    cap_Storey = False

    test_handler = ConformalContaminationTest()

    with open(savename+".txt", "r") as file:
        lines = file.readlines()
        n = eval(lines[4].split(" ")[1][:-1])
        m = eval(lines[5].split(" ")[1][:-1])
        pi_model = lines[10].split(" ")[-1][:-1]
        assert lines[4].split(" ")[0][0] == "n", ""
        assert lines[10].split(" ")[0] == "pi"

    in_data = np.load(f"{savename}.npz")

    pi_true = in_data["pi_true"]
    pvals_all = in_data["pvals_all"]
    pcon_total_ = in_data["pcon_total"]
    pi_th_arr = in_data["pi_th_arr"]
    sims = pi_true.shape[0]
    len_pi_th_arr = len(pi_th_arr)

    if (pi_model == "discrete_uniform") or (pi_model == "uniform"):
        # pi_th_arr = np.array([0])
        # len_pi_th_arr = len(pi_th_arr)
        # pi_true = pi_true[:, 0, :]
        pvals = pvals_all.reshape((sims, K-1, m))
    else:
        pvals = pvals_all.reshape((sims, len_pi_th_arr, K-1, m))

    lambda_ = n//8 / (n+1)
    i0_param = 3
    i0_arr = [m//i0_param for pi_th in pi_th_arr]

    # =============================================================================
    # Compute conformal data contamination p-values
    # =============================================================================
    pcon_total = np.zeros((sims, len_pi_th_arr, K-1, 4), dtype=np.float32)
    for sim in range(sims):
        if (sim % 200 == 0) and (sim != 0):
            print(f"Simulation: {sim:d}/{sims:d}")
        pcon = np.zeros((len_pi_th_arr, K-1, 4), dtype=np.float32)
        for pi_th_idx, pi_th in enumerate(pi_th_arr):
            for k in range(1, K):
                if (pi_model == "discrete_uniform") or (pi_model == "uniform"):
                    pvals_in = pvals[sim, k-1, :]
                else:
                    pvals_in = pvals[sim, pi_th_idx, k-1, :]
                pcon[pi_th_idx, k-1, 0], pcon[pi_th_idx, k-1, 1], pcon[pi_th_idx, k-1, 2], pcon[pi_th_idx, k-1, 3] \
                    = test_handler.all_conformal_contamination_tests_from_conformal_pvalues(pvals_in,
                      pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0_arr[pi_th_idx])
                pcon_total[sim] = pcon
    # assert np.allclose(pcon_total, pcon_total_), ""

    # =============================================================================
    # Run Storey's Benjamini-Hochberg procedure
    # =============================================================================
    FDP_total = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.float32)
    TDP_total = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.float32)
    for sim in range(sims):
        if (sim % 500 == 0) and (sim != 0):
            print(f"Simulation: {sim:d}/{sims:d}")
        for idx_alpha, alpha in enumerate(significance_levels):
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for idx in range(4):
                    if (pi_model == "discrete_uniform") or (pi_model == "uniform"):
                        pvals = pcon_total[sim, 0, :, idx]
                        pi_k = pi_true[sim, 0, :]
                    else:
                        pvals = pcon_total[sim, pi_th_idx, :, idx]
                        pi_k = pi_true[sim, pi_th_idx, :]
                    if use_true is True:
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0, K-1)
                    else:
                        _, K0_hat = Storeys_correction(gamma, pvals, K-1, cap_Storey)
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)
                    R = np.sum(rejectBool_)
                    # FDP
                    RcapH = np.sum(rejectBool_[pi_k<=pi_th])
                    if R == 0:
                        FDP = 0
                    else:
                        FDP = RcapH/R
                    FDP_total[sim, pi_th_idx, idx_alpha, idx] = FDP
                    # TDP
                    K1P = np.sum(pi_k>pi_th)
                    RcapHc = np.sum(rejectBool_[pi_k>pi_th])
                    if K1P == 0:
                        TDP = 0
                    else:
                        TDP = RcapHc/K1P
                    TDP_total[sim, pi_th_idx, idx_alpha, idx] = TDP
    FDR = np.mean(FDP_total, axis=0)
    TDR = np.mean(TDP_total, axis=0)

    # =============================================================================
    # FDR curve and table
    # =============================================================================
    for pi_th_idx, pi_th in enumerate(pi_th_arr):
        _ = plt.figure(figsize=fsize)
        for idx in range(4):
            plt.plot(significance_levels, FDR[pi_th_idx, :, idx], color=color_list[idx], label=name_list[idx])
        plt.plot(significance_levels, significance_levels, "-k")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("FDR")
        plt.title(f"{pi_th:.2f}")
        plt.legend()
        plt.show()

    # Table
    headers = [""] + list(pi_th_arr)
    table = list()
    for idx in range(4):
        table.append([f"{name_list[idx]}"] + [FDR[pi_th_idx, 2, idx] for pi_th_idx in range(len_pi_th_arr)])
        # table.append([""] + [FDP_std[pi_th_idx, 2, idx] for pi_th_idx in range(len_pi_th_arr)])
    print(tabulate(table, headers, tablefmt="latex"))

    # =============================================================================
    # TDR curve and table
    # =============================================================================
    if (pi_model != "fixed") and (pi_model != "fixed_balanced"):
        # TDR curves
        for pi_th_idx, pi_th in enumerate(pi_th_arr):
            _ = plt.figure(figsize=fsize)
            for idx in range(4):
                plt.plot(significance_levels, TDR[pi_th_idx, :, idx], color=color_list[idx], label=name_list[idx])
            # plt.plot(reject_threshold_arr, reject_threshold_arr, "-k")
            plt.ylabel("TDR")
            plt.xlabel(r"$\alpha$")
            plt.title(f"{pi_th:.2f}")
            plt.legend()
            plt.show()

        # Table
        headers = [""] + list(pi_th_arr)
        table = list()
        for idx in range(4):
            table.append([f"{name_list[idx]}"] + [TDR[pi_th_idx, 4, idx] for pi_th_idx in range(len_pi_th_arr)])
            # table.append([""] + [FDP_std[pi_th_idx, 2, idx] for pi_th_idx in range(len_pi_th_arr)])
        print(tabulate(table, headers, tablefmt="latex"))

    # =============================================================================
    # Distribution of conformal p-values of OOD samples
    # =============================================================================
    indicator_all = in_data["indicator_all"]
    outlier_colors_list = ["tab:green", "tab:orange", "tab:purple"]
    outlier_cdf_all = list()

    pvals = pvals_all
    ind = indicator_all
    outliers = pvals[ind]

    outlier_mass = np.zeros(n+1)
    for j in range(1, n+2):
        outlier_mass[j-1] = sum(np.where(np.isclose(outliers, np.ones(len(outliers))*j/(n+1)), 1, 0))
    outlier_mass = outlier_mass/np.sum(outlier_mass)
    outlier_cdf_all.append(np.cumsum(outlier_mass))

    _ = plt.figure(figsize=fsize)
    plt.plot(np.arange(1,n+2,1)/(n+1), outlier_cdf_all[0], color="tab:blue")
    plt.ylabel("Cumulative distribution function")
    plt.xlabel("Conformal p-value")
    plt.show()


    # =============================================================================
    # Distribution of conformal data contamination p-values
    # =============================================================================
    if (pi_model == "discrete_uniform") or (pi_model == "fixed"):
        if (pi_model == "discrete_uniform"):
            pi_true_flat = pi_true[:, 0, :].flatten()
        else:
            pi_true_flat = pi_true.flatten()
        pcon_flat = pcon_total.reshape(-1, 4)
        _ = plt.figure(figsize=fsize)
        pi = 0.0
        ind = np.logical_and(pi_true_flat >= pi-0.001, pi_true_flat < pi+0.001)
        CCTest_ecdf_list = list()
        for idx2 in range(4):
            y = pcon_flat[ind, idx2]
            CCTest_ecdf_list.append(ecdf(y))
            ecdf(y).cdf.plot(linestyle="solid", color=color_list[idx2], label=f"{name_list[idx2]}")

        plt.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11), "-k")
        plt.ylabel("Cumulative distribution function")
        plt.xlabel("Conformal contamination test p-value")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

