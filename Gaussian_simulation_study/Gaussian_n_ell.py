#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 08:47:44 2025

Script to do various ablation studies for the conformal contamination tests.
Make Figure S3b.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys, os
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))

from ConformalContaminationTestModule import ConformalContaminationTest


def ablation(mu0, sigma0, mu1, sigma1, n, m, pi, pi_th, alpha, lambda_, i0, test_handler):
    """
    """
    XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
    m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
    XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
    XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
    XT = np.concatenate((XT0, XT1), axis=0)

    SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)
    ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)

    pstorey, pquantile, plinear, pfisher \
        = test_handler.all_conformal_contamination_tests(
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

    return rejectBool_storey, rejectBool_quantile, rejectBool_fisher, rejectBool_linear


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    mpl.rcParams['figure.dpi'] = 600

    np.random.seed(42)
    load = True
    save_res = False

    loadname = "n_ell_pith0.1_pi0.3_m100_alpha0.05_mu14_lambda0.059_i076"
    # loadname = "n_ell_pith0.1_pi0.3_m100_alpha0.05_mu14"

    if load == False:
        sims = 10000

        mu0 = 0
        sigma0 = 1
        mu1 = 2
        sigma1 = 1

        m = 100
        pi = 0.3
        pi_th = 0.1
        alpha = 0.05

        i0_param = 3
        i0 = int(m // i0_param)
        lambda_param = 8
        lambda_ = 100//lambda_param / (100+1)
        savename = f"Ablation/n_ell_pith{pi_th:.1f}_pi{pi:.1f}_m{m:d}_alpha{alpha:.2f}_mu1{mu1:d}_lambda{lambda_:.3f}_i0{i0:d}"

        test_handler = ConformalContaminationTest()

        n_arr = np.arange(10, 200, 1)
        len_n = len(n_arr)
        rejectBool = np.zeros((sims, len_n, 4), dtype=bool)
        for sim in range(sims):
            print("Simulation number:", sim)
            for n_idx, n in enumerate(n_arr):
                # lambda_ = max(0.029, n//32 / (n+1))
                lambda_ = max(100//lambda_param / (100+1), n//lambda_param / (n+1))
                rejectBool[sim, n_idx, 0], rejectBool[sim, n_idx, 1], \
                rejectBool[sim, n_idx, 2], rejectBool[sim, n_idx, 3]  \
                    = ablation(mu0, sigma0, mu1, sigma1, n, m, pi, pi_th,
                               alpha, lambda_, i0, test_handler)
        power = np.mean(rejectBool, axis=0)
        np.savez(f"{savename}.npz",
                 n_arr=n_arr, power=power)
    elif load == True:
        in_data = np.load(f"Ablation/{loadname}.npz")
        n_arr = in_data["n_arr"]
        power = in_data["power"]

        fig = plt.figure(figsize=fsize)
        plt.plot(n_arr, power[:, 0], color="tab:orange", label="Storey")
        plt.plot(n_arr, power[:, 1], color="tab:purple", label="Quantile")
        plt.plot(n_arr, power[:, 2], color="tab:blue", label="Fisher")
        plt.plot(n_arr, power[:, 3], color="tab:green", label="Sum")
        plt.xlabel("n")
        plt.ylabel("Power")
        plt.legend()
        if save_res is True:
            plt.savefig(f"Ablation/{loadname}.png", dpi=500, bbox_inches="tight")
        plt.show()

        if save_res is True:
            with open(f"Ablation/{loadname}.txt", "w") as file:
                name_list = ["Storey", "Quantile", "Fisher", "Linear"]
                color_list = ["color2", "color3", "color4", "color5", "color6"]
                marker_list = ["square", "diamond", "star", "triangle"]
                linestyle_list = ["solid", "dashed", "dotted", "dashdotted"]
                for idx in range(4):
                    file.write(f"\\addplot[semithick, mark={marker_list[idx]}"+", mark options={solid},"+f" {linestyle_list[idx]}, {color_list[idx]}]\n")
                    file.write("table{%\n")
                    for x, y in zip(n_arr[::3], power[::3, idx]):
                        file.write(f"{x:d}  {y:.4f}\n")
                    # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
                    file.write("};\\label{plot:"+f"{name_list[idx]}"+"}\n")





