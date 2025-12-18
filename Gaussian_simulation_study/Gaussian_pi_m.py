#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:08:44 2025

Script to do various ablation studies for the conformal contamination tests.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from scipy.stats import binom as ssbinom

import sys, os
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))

from ConformalContaminationTestModule import ConformalContaminationTest


def ablation_pi(mu0, sigma0, mu1, sigma1, n, m, pi, pi_th, alpha, lambda_, i0, test_handler):
    """
    """
    # Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(ell, 2)).astype(np.float32)
    XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
    m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
    XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
    XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
    XT = np.concatenate((XT0, XT1), axis=0)

    # OCSVM = OneClassSVM()
    # OCSVM.fit(Xtrain)
    # SC = OCSVM.score_samples(XC).astype(np.float32)
    # ST = OCSVM.score_samples(XT).astype(np.float32)
    SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)
    ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)

    pstorey, pquantile, _, plinear, pfisher \
        = test_handler.almost_all_conformal_contamination_tests(
            SC, ST, pi_th=pi_th,n=n, m=m, lambda_=lambda_, i0=i0)

    if pstorey <= alpha:
        rejectBool_storey = True
    else:
        rejectBool_storey = False

    if pquantile <= alpha:
        rejectBool_quantile = True
    else:
        rejectBool_quantile = False

    if pfisher <= alpha:
        rejectBool_fisher = True
    else:
        rejectBool_fisher = False

    if plinear <= alpha:
        rejectBool_linear = True
    else:
        rejectBool_linear = False

    return rejectBool_storey, rejectBool_quantile, rejectBool_fisher, rejectBool_linear, pstorey, pquantile, plinear, pfisher


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)

    np.random.seed(42)
    load = True

    if load == False:
        sims = 1000

        mu0 = 0
        sigma0 = 1
        mu1 = 4
        sigma1 = 1

        # ell = 100
        n = 100
        pi_th = 0.1
        alpha = 0.05

        lambda_ = n//16 / (n+1)
        i0_param = 1.3

        test_handler = ConformalContaminationTest()
        savename = f"Ablation/pi_m_pith{pi_th:.1f}_n{n:d}_alpha{alpha:.2f}_mu1{mu1:d}_lambda{lambda_:.3f}_i0{i0_param:.2f}"

        m_arr = np.linspace(100, 800, 5).astype(np.int32)
        pi_arr = pi_th + np.linspace(1e-03, 0.2, 20)
        len_pi, len_m = len(pi_arr), len(m_arr)
        rejectBool = np.zeros((sims, len_pi, len_m, 4), dtype=bool)
        pvalue = np.zeros((sims, len_pi, len_m, 4), dtype=bool)
        for sim in range(sims):
            print("Simulation number:", sim)
            for pi_idx, pi in enumerate(pi_arr):
                for m_idx, m in enumerate(m_arr):
                    # i0 = int(m-ssbinom.ppf(0.998, m, pi_th))
                    i0 = int(m // i0_param)
                    rejectBool[sim, pi_idx, m_idx, 0], rejectBool[sim, pi_idx, m_idx, 1], \
                    rejectBool[sim, pi_idx, m_idx, 2], rejectBool[sim, pi_idx, m_idx, 3], \
                    pvalue[sim, pi_idx, m_idx, 0], pvalue[sim, pi_idx, m_idx, 1], \
                    pvalue[sim, pi_idx, m_idx, 2], pvalue[sim, pi_idx, m_idx, 3], \
                        = ablation_pi(mu0, sigma0, mu1, sigma1, n, m, pi, pi_th, alpha, lambda_, i0, test_handler)
        power = np.mean(rejectBool, axis=0)
        np.savez(f"{savename}.npz",
                 pi_arr=pi_arr, m_arr=m_arr, power=power)
    elif load == True:
        in_data = np.load("Ablation/pi_m_pith0.1_n100_ell100_alpha0.05_mu14_lambda0.079_i01.50.npz")
        pi_arr = in_data["pi_arr"]
        m_arr = in_data["m_arr"]
        power = in_data["power"]

        # linestyle_list = ["solid", "dashed", "dashdot", "dotted"]
        color_list = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green"]
        fig = plt.figure(figsize=fsize)
        for m_idx, m in enumerate(m_arr):
            # DeltaPi = (pi_arr - pi_th)*m
            # plt.plot(DeltaPi, power[:, m_idx, 0], linestyle=linestyle_list[m_idx], color="tab:orange", label="Storey")
            # plt.plot(DeltaPi, power[:, m_idx, 1], linestyle=linestyle_list[m_idx], color="tab:purple", label="Quantile")
            # plt.plot(DeltaPi, power[:, m_idx, 2], linestyle=linestyle_list[m_idx], color="tab:blue", label=f"Fisher {m:.2f}")
            # plt.plot(DeltaPi, power[:, m_idx, 3], linestyle=linestyle_list[m_idx], color="tab:green", label="Summation")
            # plt.plot(pi_arr, power[:, m_idx, 2], linestyle=linestyle_list[m_idx], color="tab:blue", label=f"Fisher {m:.2f}")
            plt.plot(pi_arr, power[:, m_idx, 0], color=color_list[m_idx], linestyle="solid", label=f"Storey (m = {m:d})")
            plt.plot(pi_arr, power[:, m_idx, 1], color=color_list[m_idx], linestyle="dashed", label=f"Quantile (m = {m:d})")
            plt.plot(pi_arr, power[:, m_idx, 2], color=color_list[m_idx], linestyle="dotted", label=f"Fisher (m = {m:d})")
            plt.plot(pi_arr, power[:, m_idx, 3], color=color_list[m_idx], linestyle="dashdot", label=f"Linear (m = {m:d})")

        # plt.xlabel(r"$\Delta_m\pi$")
        plt.xlabel(r"$\pi$")
        plt.ylabel("Power")
        plt.legend()
        # plt.savefig("Ablation/pi_m_pith0.1_n100_ell100_alpha0.05_mu14.png", dpi=500, bbox_inches="tight")
        plt.show()

        with open("Ablation/pi_m_pith0.1_n100_ell100_alpha0.05_mu14_lambda0.079_i01.50.txt", "w") as file:
            name_list = ["Storey", "Quantile", "Fisher", "Linear"]
            color_list = ["color2", "color3", "color4", "color5", "color6"]
            marker_list = ["square", "diamond", "star", "triangle"]
            linestyle_list = ["solid", "dashed", "dotted", "dashdotted"]
            for idx in range(4):
                for m_idx, m in enumerate(m_arr):
                    file.write(f"\\addplot[semithick, mark={marker_list[idx]}"+", mark options={solid},"+f" {linestyle_list[idx]}, {color_list[m_idx]}]\n")
                    file.write("table{%\n")
                    for x, y in zip(pi_arr, power[:, m_idx, idx]):
                        file.write(f"{x:.4f}  {y:.4f}\n")
                    # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
                    file.write("};\\label{plot:"+f"{name_list[idx]}_m" + f"{m:d}" + "}\n")

