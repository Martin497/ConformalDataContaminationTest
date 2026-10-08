#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:24:40 2025

In this script we do testing of the null hypothesis H_0 : \pi \leq \pi_th
using the proposed test statistics.
The data in consideration is Gaussian.
Make Figure S4.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys, os
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))

from ConformalContaminationTestModule import ConformalContaminationTest
from Benjamini_Hochberg import Benjamini_Hochberg_procedure, compute_FDR, compute_TDR, Storeys_correction


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    mpl.rcParams['figure.dpi'] = 600

    np.random.seed(49)
    load = True
    save_res = False

    loadname = "mu4_n200_m100_pinull0.20_pialt0.30_K20_K010_lambda0.030_i066_gamma0.50"
    color_list = ["tab:blue", "tab:purple", "tab:orange", "tab:green"]


    if load == False:
        sims = 2000

        mu0 = 0
        sigma0 = 1
        mu1 = 3
        sigma1 = 1

        # n_train = 200
        n = 200

        K = 20
        K0 = 10
        m = 100

        pi_null = 0.2
        pi_alt = 0.3
        pi_th = 0.2
        assert pi_null <= pi_th and pi_alt > pi_th

        alpha_arr = np.linspace(0.001, 0.5, 100)
        # lambda_ = n//32 / (n+1)
        # i0 = int(m-ssbinom.ppf(0.998, m, pi_th))
        i0_param = 1.5
        i0 = int(m // i0_param)
        lambda_param = 32
        lambda_ = 100//lambda_param / (100+1)
        B = 1000
        gamma = 0.5

        test_handler = ConformalContaminationTest()

        rejectBool_storey = np.zeros((sims, len(alpha_arr), K), dtype=bool)
        rejectBool_quantile = np.zeros((sims, len(alpha_arr), K), dtype=bool)
        rejectBool_fisher = np.zeros((sims, len(alpha_arr), K), dtype=bool)
        rejectBool_linear = np.zeros((sims, len(alpha_arr), K), dtype=bool)
        # rejectBool_shiftFisher = np.zeros((sims, len(alpha_arr), K), dtype=bool)
        for i in range(sims):
            print(i+1, "/", sims)
            # Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
            XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)

            # OCSVM = OneClassSVM()
            # OCSVM.fit(Xtrain)
            # SC = OCSVM.score_samples(XC).astype(np.float32)
            SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)

            p_hat_storey = np.zeros(K, dtype=np.float32)
            p_hat_quantile = np.zeros(K, dtype=np.float32)
            p_hat_fisher = np.zeros(K, dtype=np.float32)
            p_hat_linear = np.zeros(K, dtype=np.float32)
            p_hat_shiftFisher = np.zeros(K, dtype=np.float32)
            for k in range(K):
                if k < K0:
                    m0 = np.random.binomial(m, p=1-pi_null)
                else:
                    m0 = np.random.binomial(m, p=1-pi_alt)
                XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
                XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
                XT = np.concatenate((XT0, XT1), axis=0)

                # ST = OCSVM.score_samples(XT).astype(np.float32)
                ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)

                # pmarg_all = np.zeros(m, dtype=np.float32)
                # for j in range(m):
                #     pmarg_all[j] = (1 + np.sum(SC <= ST[j]))/(n+1)
                # Tstorey = round(np.sum(pmarg_all > lambda_))
                # Tquantile = round(np.sort(pmarg_all)[m-i0-1] * (n+1))
                # Tfisher = -2*np.sum(np.log((n+2)/(n+1) - pmarg_all))
                # p_hat_storey[k] = p_hat_storey_lookup[Tstorey]
                # p_hat_quantile[k] = p_hat_quantile_lookup[Tquantile]
                # p_hat_fisher[k] = Fisher_pvalue(pi_th, n, m, Tfisher)

                p_hat_storey[k], p_hat_quantile[k], p_hat_linear[k], p_hat_fisher[k] \
                    = test_handler.all_conformal_contamination_tests(SC, ST, pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0, B=B)

            # K0_hat = K0
            _, K0_hat_Storey = Storeys_correction(gamma, p_hat_storey, K)
            _, K0_hat_Quantile = Storeys_correction(gamma, p_hat_quantile, K)
            _, K0_hat_Fisher = Storeys_correction(gamma, p_hat_fisher, K)
            _, K0_hat_Linear = Storeys_correction(gamma, p_hat_linear, K)
            for alpha_idx, alpha in enumerate(alpha_arr):
                rejectBool_storey[i, alpha_idx] = Benjamini_Hochberg_procedure(p_hat_storey, alpha, K0_hat_Storey, K)
                rejectBool_quantile[i, alpha_idx] = Benjamini_Hochberg_procedure(p_hat_quantile, alpha, K0_hat_Quantile, K)
                rejectBool_fisher[i, alpha_idx] = Benjamini_Hochberg_procedure(p_hat_fisher, alpha, K0_hat_Fisher, K)
                rejectBool_linear[i, alpha_idx] = Benjamini_Hochberg_procedure(p_hat_linear, alpha, K0_hat_Linear, K)

        FDR_storey = np.zeros(len(alpha_arr), dtype=np.float32)
        FDR_quantile = np.zeros(len(alpha_arr), dtype=np.float32)
        FDR_fisher = np.zeros(len(alpha_arr), dtype=np.float32)
        FDR_linear = np.zeros(len(alpha_arr), dtype=np.float32)
        for alpha_idx, alpha in enumerate(alpha_arr):
            FDR_storey[alpha_idx], _ = compute_FDR(rejectBool_storey[:, alpha_idx, :], K0, sims)
            FDR_quantile[alpha_idx], _ = compute_FDR(rejectBool_quantile[:, alpha_idx, :], K0, sims)
            FDR_fisher[alpha_idx], _ = compute_FDR(rejectBool_fisher[:, alpha_idx, :], K0, sims)
            FDR_linear[alpha_idx], _ = compute_FDR(rejectBool_linear[:, alpha_idx, :], K0, sims)

        TDR_storey = np.zeros(len(alpha_arr), dtype=np.float32)
        TDR_quantile = np.zeros(len(alpha_arr), dtype=np.float32)
        TDR_fisher = np.zeros(len(alpha_arr), dtype=np.float32)
        TDR_linear = np.zeros(len(alpha_arr), dtype=np.float32)
        for alpha_idx, alpha in enumerate(alpha_arr):
            TDR_storey[alpha_idx], _ = compute_TDR(rejectBool_storey[:, alpha_idx, :], K0, K, sims)
            TDR_quantile[alpha_idx], _ = compute_TDR(rejectBool_quantile[:, alpha_idx, :], K0, K, sims)
            TDR_fisher[alpha_idx], _ = compute_TDR(rejectBool_fisher[:, alpha_idx, :], K0, K, sims)
            TDR_linear[alpha_idx], _ = compute_TDR(rejectBool_linear[:, alpha_idx, :], K0, K, sims)

        # print(f"False discovery rate) Storey: {FDR_storey:.4f}    Quantile: {FDR_quantile:.4f}    Fisher: {FDR_fisher:.4f}")
        # print(f" True discovery rate) Storey: {TDR_storey:.4f}    Quantile: {TDR_quantile:.4f}    Fisher: {TDR_fisher:.4f}")

        savename = f"mu{mu1:d}_n{n:d}_m{m:d}_pinull{pi_null:.2f}_pialt{pi_alt:.2f}_K{K:d}_K0{K0:d}_lambda{lambda_:.3f}_i0{i0:d}_gamma{gamma:.2f}"

        np.savez(f"Multiple/{savename}.npz", alpha_arr=alpha_arr, FDR_storey=FDR_storey, FDR_quantile=FDR_quantile,
                 FDR_fisher=FDR_fisher, FDR_linear=FDR_linear, TDR_storey=TDR_storey, TDR_quantile=TDR_quantile,
                 TDR_fisher=TDR_fisher, TDR_linear=TDR_linear)

        fig = plt.figure(figsize=fsize)
        plt.plot(alpha_arr, FDR_storey, color=color_list[0], label="Storey")
        plt.plot(alpha_arr, FDR_quantile, color=color_list[1], label="quantile")
        plt.plot(alpha_arr, FDR_fisher, color=color_list[2], label="Fisher")
        plt.plot(alpha_arr, FDR_linear, color=color_list[3], label="Sum")
        plt.plot(alpha_arr, alpha_arr, color="k")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("FDR")
        plt.legend()
        plt.savefig(f"Multiple/FDR_{savename}.png", bbox_inches="tight", dpi=500)
        plt.show()

        fig = plt.figure(figsize=fsize)
        plt.plot(alpha_arr, TDR_storey, color=color_list[0], label="Storey")
        plt.plot(alpha_arr, TDR_quantile, color=color_list[1], label="quantile")
        plt.plot(alpha_arr, TDR_fisher, color=color_list[2], label="Fisher")
        plt.plot(alpha_arr, TDR_linear, color=color_list[3], label="Sum")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("TDR")
        plt.legend()
        plt.savefig(f"Multiple/TDR_{savename}.png", bbox_inches="tight", dpi=500)
        plt.show()

    else:
        in_data = np.load(f"Multiple/{loadname}.npz")
        alpha_arr = in_data["alpha_arr"]
        FDR_storey = in_data["FDR_storey"]
        FDR_quantile = in_data["FDR_quantile"]
        FDR_fisher = in_data["FDR_fisher"]
        FDR_linear = in_data["FDR_linear"]
        TDR_storey = in_data["TDR_storey"]
        TDR_quantile = in_data["TDR_quantile"]
        TDR_fisher = in_data["TDR_fisher"]
        TDR_linear = in_data["TDR_linear"]

        fig = plt.figure(figsize=fsize)
        plt.plot(alpha_arr, FDR_storey, color=color_list[0], label="Storey")
        plt.plot(alpha_arr, FDR_quantile, color=color_list[1], label="quantile")
        plt.plot(alpha_arr, FDR_fisher, color=color_list[2], label="Fisher")
        plt.plot(alpha_arr, FDR_linear, color=color_list[3], label="Sum")
        plt.plot(alpha_arr, alpha_arr, color="k")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("FDR")
        plt.legend()
        if save_res is True:
            plt.savefig(f"Multiple/FDR_{loadname}.png", bbox_inches="tight", dpi=500)
        plt.show()

        fig = plt.figure(figsize=fsize)
        plt.plot(alpha_arr, TDR_storey, color=color_list[0], label="Storey")
        plt.plot(alpha_arr, TDR_quantile, color=color_list[1], label="quantile")
        plt.plot(alpha_arr, TDR_fisher, color=color_list[2], label="Fisher")
        plt.plot(alpha_arr, TDR_linear, color=color_list[3], label="Sum")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("TDR")
        plt.legend()
        if save_res is True:
            plt.savefig(f"Multiple/TDR_{loadname}.png", bbox_inches="tight", dpi=500)
        plt.show()

        if save_res is True:
            with open(f"Multiple/FDR_{loadname}.txt", "w") as file:
                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color2]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, FDR_storey):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color3]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, FDR_quantile):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color4]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, FDR_fisher):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color5]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, FDR_linear):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

            with open(f"Multiple/TDR_{loadname}.txt", "w") as file:
                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color2]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, TDR_storey):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color3]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, TDR_quantile):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color4]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, TDR_fisher):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")

                file.write("\\addplot[semithick, mark=square, mark options={solid}, solid, color5]\n")
                file.write("table{%\n")
                for x, y in zip(alpha_arr, TDR_linear):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("};\n")