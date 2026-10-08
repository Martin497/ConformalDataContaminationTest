#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:06:05 2025

In this script we do testing of the null hypothesis H_0 : pi \leq pi_th
using the proposed test statistics.
The data in consideration is Gaussian.
"""


import numpy as np
import matplotlib.pyplot as plt

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from ConformalContaminationTestModule import ConformalContaminationTest


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    np.random.seed(42)
    inloop_plotting = False
    save_res = False

    # =============================================================================
    # Setup and simulate
    # =============================================================================
    sims = 10000

    mu0 = 0
    sigma0 = 1
    sigma1 = 1

    n = 200
    m = 50

    pi = 0.7
    pi_th = 0.5

    alpha = 0.05

    mu1_arr = np.array([2])
    mu1_idx = 0

    lambda_arr = np.arange(0, n+2, 1) / (n+1)
    i0_arr = np.arange(0, m-1, 1)
    B = 1

    len_mu1 = len(mu1_arr)
    len_lambda = len(lambda_arr)
    len_i0 = len(i0_arr)

    test_handler = ConformalContaminationTest()

    # =============================================================================
    # Make p-value look-up tables
    # =============================================================================
    print("Making look-up tables...")
    p_hat_storey_lookup = np.zeros((len_lambda, m+1), dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
        for j in range(m+1):
            p_hat_storey_lookup[lambda_idx, j] = test_handler.Storey_pvalue(j, pi_th=pi_th, n=n, m=m, lambda_=lambda_)
    p_hat_quantile_lookup = np.zeros((len_i0, n+2), dtype=np.float64)
    for i0_idx, i0 in enumerate(i0_arr):
        for j in range(n+2):
            p_hat_quantile_lookup[i0_idx, j] = test_handler.quantile_pvalue(j, pi_th=pi_th, n=n, m=m, i0=i0)

    print("Simulating data...")
    # =============================================================================
    # Simulate data
    # =============================================================================
    p_marg_all_arr = np.zeros((sims, len_mu1, m), dtype=np.float64)
    m0_arr = np.zeros(sims, dtype=np.int32)
    for sim_idx in range(sims):
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
        SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)

        m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
        m0_arr[sim_idx] = m0
        XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
        for mu1_idx, mu1 in enumerate(mu1_arr):
            XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
            XT = np.concatenate((XT0, XT1), axis=0)
            ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)
            p_marg_all_arr[sim_idx, mu1_idx] = test_handler.compute_conformal_pvalues(SC, ST)

    # =============================================================================
    # Estimate rejection rate
    # =============================================================================
    print("Computing empirical rejection rate...")
    rejectBool_storey = np.zeros((sims, len_lambda, len_mu1), dtype=bool)
    rejectBool_quantile = np.zeros((sims, len_i0, len_mu1), dtype=bool)
    rejectBool_fisher = np.zeros((sims, len_mu1), dtype=bool)
    rejectBool_linear = np.zeros((sims, len_mu1), dtype=bool)
    TestStat_storey = np.zeros((sims, len_lambda, len_mu1), dtype=np.int32)
    TestStat_quantile = np.zeros((sims, len_i0, len_mu1), dtype=np.int32)
    TestStat_fisher = np.zeros((sims, len_i0, len_mu1), dtype=np.int32)
    for sim_idx in range(sims):
        if sim_idx % 500 == 0:
            print("Simulation", sim_idx, "/", sims)
        ### Storey loop ###
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            for mu1_idx, mu1 in enumerate(mu1_arr):
                test_stat = test_handler.Storey_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], lambda_=lambda_)
                TestStat_storey[sim_idx, lambda_idx, mu1_idx] = test_stat
                if p_hat_storey_lookup[lambda_idx, test_stat] <= alpha:
                    rejectBool_storey[sim_idx, lambda_idx, mu1_idx] = True
        ### Quantile loop ###
        for i0_idx, i0 in enumerate(i0_arr):
            for mu1_idx, mu1 in enumerate(mu1_arr):
                test_stat = test_handler.quantile_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], n=n, i0=i0)
                TestStat_quantile[sim_idx, i0_idx, mu1_idx] = test_stat
                if p_hat_quantile_lookup[i0_idx, test_stat] <= alpha:
                    rejectBool_quantile[sim_idx, i0_idx, mu1_idx] = True
        ### Fisher ###
        for mu1_idx, mu1 in enumerate(mu1_arr):
            test_stat = test_handler.Fisher_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], n=n)
            TestStat_fisher[sim_idx, mu1_idx] = test_stat
            pvalue = test_handler.Fisher_pvalue(test_stat, pi_th=pi_th, n=n, m=m)
            if pvalue <= alpha:
                rejectBool_fisher[sim_idx, mu1_idx] = True
        ### Linear ###
        for mu1_idx, mu1 in enumerate(mu1_arr):
            test_stat = test_handler.linear_test_statistic(p_marg_all_arr[sim_idx, mu1_idx])
            pvalue = test_handler.linear_pvalue(test_stat, pi_th=pi_th, n=n, m=m)
            if pvalue <= alpha:
                rejectBool_linear[sim_idx, mu1_idx] = True

    rejection_rate_storey = np.sum(rejectBool_storey, axis=0)/sims
    rejection_rate_quantile = np.sum(rejectBool_quantile, axis=0)/sims
    rejection_rate_fisher = np.sum(rejectBool_fisher, axis=0)/sims
    rejection_rate_linear = np.sum(rejectBool_linear, axis=0)/sims

    # =============================================================================
    # Comparison of test statistics
    # =============================================================================
    fig = plt.figure(figsize=fsize)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    l_linear = ax1.plot(np.array([0, 1]), np.array([rejection_rate_linear[mu1_idx], rejection_rate_linear[mu1_idx]]), color="tab:blue", label="Linear")
    l_fisher = ax1.plot(np.array([0, 1]), np.array([rejection_rate_fisher[mu1_idx], rejection_rate_fisher[mu1_idx]]), color="tab:green", label="Fisher")
    l_storey = ax1.plot(lambda_arr, rejection_rate_storey[:, mu1_idx], "x", color="tab:orange", label="Storey")
    l_quantile = ax2.plot(i0_arr, rejection_rate_quantile[:, mu1_idx], "x", color="tab:purple", label="quantile")
    ax1.set_xlabel(r"$\lambda$")
    ax2.set_xlabel(r"$i_0$")
    ax1.set_ylabel("Rejection rate")
    ax1.set_xlim(0, 1)
    lns = l_storey+l_quantile+l_fisher+l_linear
    ax1.legend(lns, [l.get_label() for l in lns])
    ax2.grid(linewidth=0)
    if save_res is True:
        plt.savefig(f"TestStatistic_Hyper/comparison_mu{int(mu1_arr[mu1_idx]):d}_n{n:d}_m{m:}_pi{pi:.1f}_pith{pi_th:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
    plt.show()

    if save_res is True:
        with open(f"TestStatistic_Hyper/comparison_mu{int(mu1_arr[mu1_idx]):d}_n{n:d}_m{m:}_pi{pi:.1f}_pith{pi_th:.1f}_alpha{alpha:.2f}.txt", "w") as file:
            name_list = ["Storey", "Quantile", "Fisher", "Linear"]
            color_list = ["color2", "color3", "color4", "color5", "color6"]
            marker_list = ["square", "diamond", "star", "triangle"]
            linestyle_list = ["solid", "dashed", "dotted", "dashdotted"]

            file.write(f"\\addplot[semithick, mark={marker_list[0]}"+", mark options={solid},"+f" {linestyle_list[0]}, {color_list[0]}]\n")
            file.write("table{%\n")
            for x, y in zip(lambda_arr, rejection_rate_storey[:, mu1_idx]):
                file.write(f"{x:.4f}  {y:.4f}\n")
            # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
            file.write("};\\label{plot:Storey_hyper}\n")

            file.write(f"\\addplot[semithick, mark={marker_list[0]}"+", mark options={solid},"+f" {linestyle_list[0]}, {color_list[1]}]\n")
            file.write("table{%\n")
            for x, y in zip(i0_arr, rejection_rate_quantile[:, mu1_idx]):
                file.write(f"{x:d}  {y:.4f}\n")
            # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
            file.write("};\\label{plot:Quantile_hyper}\n")

            file.write(f"\\addplot[semithick, mark={marker_list[0]}"+", mark options={solid},"+f" {linestyle_list[0]}, {color_list[2]}]\n")
            file.write("table{%\n")
            for x, y in zip(np.array([0, 1]), np.array([rejection_rate_fisher[mu1_idx], rejection_rate_fisher[mu1_idx]])):
                file.write(f"{x:d}  {y:.4f}\n")
            # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
            file.write("};\\label{plot:Fisher_hyper}\n")

            file.write(f"\\addplot[semithick, mark={marker_list[0]}"+", mark options={solid},"+f" {linestyle_list[0]}, {color_list[3]}]\n")
            file.write("table{%\n")
            for x, y in zip(np.array([0, 1]), np.array([rejection_rate_linear[mu1_idx], rejection_rate_linear[mu1_idx]])):
                file.write(f"{x:d}  {y:.4f}\n")
            # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
            file.write("};\\label{plot:Linear_hyper}\n")
