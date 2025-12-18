#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:33:38 2025

@author: martin
"""


import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from Benjamini_Hochberg import Storeys_correction, Benjamini_Hochberg_procedure


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    color_list = ["tab:red", "tab:olive", "tab:purple", "tab:orange"]
    name_list = ["Storey", "Quantile", "Summation", "Fisher"]

    savename = "retina/retina_OCSVM_discrete_uniform"

    gamma = 0.5 # Storey's hyperparameter conformal contamination test
    num_rejection_thresholds = 40
    significance_levels = np.linspace(0, 0.975, num_rejection_thresholds)

    with open(savename+".txt", "r") as file:
        lines = file.readlines()
        Kminus1 = eval(lines[2].split(" ")[1][:-1])
        K = Kminus1 + 1
        n = eval(lines[4].split(" ")[1][:-1])
        m = eval(lines[5].split(" ")[1][:-1])
        # K0 = eval(lines[-1].split(" ")[1][:-1])
        pi_model = lines[10].split(" ")[-1][:-1]
        assert lines[4].split(" ")[0][0] == "n", ""
        assert lines[10].split(" ")[0] == "pi"

    K0 = None #int((K-1)/2) # int((K-1)/2)
    use_true = False
    cap_Storey = False

    in_data = np.load(f"{savename}.npz")

    pi_true = in_data["pi_true"]
    pcon_total = in_data["pcon_total"]
    pi_th_arr = in_data["pi_th_arr"]
    len_pi_th_arr = len(pi_th_arr)
    sims = pcon_total.shape[0]

    FDP_total = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.float32)
    TDP_total = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.float32)
    for sim in range(sims):
        if (sim % 500 == 0) and (sim != 0):
            print(f"Simulation: {sim:d}/{sims:d}")
        for idx_alpha, alpha in enumerate(significance_levels):
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for idx in range(4):
                    pvals = pcon_total[sim, pi_th_idx, :, idx]
                    pi_k = pi_true[sim, pi_th_idx, :]
                    if use_true is True:
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0, K-1)
                    else:
                        _, K0_hat = Storeys_correction(gamma, pvals, K-1, cap_Storey)
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)
                    # FDP
                    R = np.sum(rejectBool_)
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

    if (pi_model == "fixed") or (pi_model == "discrete_uniform") \
    or (pi_model == "fixed_part") or (pi_model == "fixed_balanced"):
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
        print("\n FDR table\n")
        headers = [""] + list(pi_th_arr)
        table = list()
        for idx in range(4):
            table.append([f"{name_list[idx]}"] + [FDR[pi_th_idx, 4, idx] for pi_th_idx in range(len_pi_th_arr)])
            # table.append([""] + [FDP_std[pi_th_idx, 2, idx] for pi_th_idx in range(len_pi_th_arr)])
        print(tabulate(table, headers, tablefmt="latex"))

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
        print("\n TDR table\n")
        headers = [""] + list(pi_th_arr)
        table = list()
        for idx in range(4):
            table.append([f"{name_list[idx]}"] + [TDR[pi_th_idx, 4, idx] for pi_th_idx in range(len_pi_th_arr)])
            # table.append([""] + [FDP_std[pi_th_idx, 2, idx] for pi_th_idx in range(len_pi_th_arr)])
        print(tabulate(table, headers, tablefmt="latex"))


