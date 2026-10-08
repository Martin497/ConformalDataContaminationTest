#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:28:37 2025

Two-sample testing with synthetic Gaussian data.
Make Figure S2.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.svm import OneClassSVM

import sys, os
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.append(os.path.dirname(os.getcwd()))

from ConformalContaminationTestModule import ConformalContaminationTest


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    mpl.rcParams['figure.dpi'] = 600

    np.random.seed(49)
    load = True
    save = False
    save_res = False

    loadname = "TwoSample/pi_n200_m100_alpha0.05_mu14"

    if load == False:
        sims = 2000

        # n_train = 200
        n = 200
        m = 100

        mu0 = 0
        sigma0 = 1
        mu1 = 4
        sigma1 = 1
        pi_arr = np.linspace(1e-03, 0.25, 10)
        len_pi = len(pi_arr)

        pi_th = 0
        alpha = 0.05

        i0_param = 1.1
        i0 = int(m // i0_param)
        lambda_param = 32
        lambda_ = n//lambda_param / (n+1)
        # lambda_ = n//32 / (n+1)
        # i0 = m-10
        assert pi_th == 0, "We are interested in the special case pi_th = 0."

        T = 150
        permutations = 500
        permutations_train = 200

        test_handler = ConformalContaminationTest()

        rejectBoolCCstorey = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolCCquantile = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolCCfisher = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolCClinear = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolKS = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolFish = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolKSPermut = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolL2Permut = np.zeros((sims, len_pi), dtype=bool)
        rejectBoolOCSVM = np.zeros((sims, len_pi), dtype=bool)
        for i in range(sims):
            print("Simulation number:", i)
            for pi_idx, pi in enumerate(pi_arr):
                # Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
                XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
                m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
                XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
                XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
                XT = np.concatenate((XT0, XT1), axis=0)

                # =============================================================================
                # Compute conformal scores
                # =============================================================================
                # OCSVM = OneClassSVM()
                # OCSVM.fit(Xtrain)
                # SC = OCSVM.score_samples(XC).astype(np.float32)
                # ST = OCSVM.score_samples(XT).astype(np.float32)
                SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)
                ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)

                # =============================================================================
                # Conformal contamination tests
                # =============================================================================
                pstorey, pquantile, plinear, pfisher = test_handler.all_conformal_contamination_tests(
                    SC, ST, pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0)

                if pstorey <= alpha:
                    rejectBoolCCstorey[i, pi_idx] = True
                if pquantile <= alpha:
                    rejectBoolCCquantile[i, pi_idx] = True
                if pfisher <= alpha:
                    rejectBoolCCfisher[i, pi_idx] = True
                if plinear <= alpha:
                    rejectBoolCClinear[i, pi_idx] = True

                # =============================================================================
                # Compute the empirical cdfs
                # =============================================================================
                amin = min(np.min(SC), np.min(ST))
                amax = max(np.max(SC), np.max(ST))
                HC, X1 = np.histogram(SC, bins=T, range=(amin, amax),  density=True)
                SC_cdf = np.cumsum(HC)*(X1[1] - X1[0])
                HT, _ = np.histogram(ST, bins=T, range=(amin, amax), density=True)
                ST_cdf = np.cumsum(HT)*(X1[1] - X1[0])

                # =============================================================================
                # Two-sample Kolmogorov-Smirnov test
                # =============================================================================
                Tsmirnov = np.max(np.abs(SC_cdf - ST_cdf))
                rejectBoolKS[i, pi_idx] = Tsmirnov > np.sqrt(-np.log(alpha/2)*(1+m/n)/(2*m))

                # =============================================================================
                # Two-sample permutation test
                # =============================================================================
                concatenated_scores = np.hstack((SC, ST))
                conatenated_scores_sampled = np.random.choice(concatenated_scores, size=(permutations, m+n), replace=True)
                SC_permut = conatenated_scores_sampled[:, :n]
                ST_permut = conatenated_scores_sampled[:, n:]
                SC_permut_cdf = np.zeros((permutations, T))
                ST_permut_cdf = np.zeros((permutations, T))
                for j in range(permutations):
                    HC_permut, _ = np.histogram(SC_permut[j], bins=T, range=(amin, amax),  density=True)
                    SC_permut_cdf[j] = np.cumsum(HC_permut)*(X1[1] - X1[0])
                    HT_permut, _ = np.histogram(ST_permut[j], bins=T, range=(amin, amax), density=True)
                    ST_permut_cdf[j] = np.cumsum(HT_permut)*(X1[1] - X1[0])

                ### Kolmogorov-Smirnov test statistic ###
                Tsmirnov_permut = np.max(np.abs(SC_permut_cdf - ST_permut_cdf), axis=1)
                p_KS_permut = (1 + np.sum(Tsmirnov_permut >= Tsmirnov))/(permutations+1)
                rejectBoolKSPermut[i, pi_idx] = p_KS_permut <= alpha

                ### L2 test statistic ###
                Tl2 = np.mean(np.abs(SC_cdf - ST_cdf)**2)
                Tl2_permut = np.mean(np.abs(SC_permut_cdf - ST_permut_cdf)**2, axis=1)
                p_l2_permut = (1 + np.sum(Tl2_permut >= Tl2))/(permutations+1)
                rejectBoolL2Permut[i, pi_idx] = p_l2_permut <= alpha

                ### More general statistics ###
                Tdiff0 = SC_cdf - ST_cdf
                Tdiff_permut = SC_permut_cdf - ST_permut_cdf
                Tdiff_permut_train = Tdiff_permut[:permutations_train]
                Tdiff_permut_null = Tdiff_permut[permutations_train:]
                Tdiff = np.concatenate((np.expand_dims(Tdiff0, axis=0), Tdiff_permut_null), axis=0)

                ### Approximate projection depth ###
                # TDD = projection(Tdiff, Tdiff_permut_train)
                # rejectBoolDD[i] = (1 + np.sum(TDD[1:] <= TDD[0]))/(permutations-permutations_train+1) <= alpha

                ### One-class support vector machines ###
                OCSVM = OneClassSVM()
                OCSVM.fit(Tdiff_permut[:permutations_train])
                TOCSVM = OCSVM.score_samples(Tdiff).astype(np.float32)
                rejectBoolOCSVM[i, pi_idx] = (1 + np.sum(TOCSVM[1:] <= TOCSVM[0]))/(permutations-permutations_train+1) <= alpha

        powerCCstorey = np.mean(rejectBoolCCstorey, axis=0)
        powerCCquantile = np.mean(rejectBoolCCquantile, axis=0)
        powerCCfisher = np.mean(rejectBoolCCfisher, axis=0)
        powerCClinear = np.mean(rejectBoolCClinear, axis=0)
        powerKS = np.mean(rejectBoolKS, axis=0)
        powerKSPermut = np.mean(rejectBoolKSPermut, axis=0)
        powerL2Permut = np.mean(rejectBoolL2Permut, axis=0)
        powerOCSVM = np.mean(rejectBoolOCSVM, axis=0)

        if save is True:
            np.savez(f"TwoSample/pi_n{n:d}_m{m:d}_alpha{alpha:.2f}_mu1{mu1:d}_lambda{lambda_:.3f}_i0{i0:d}.npz",
                      pi_arr=pi_arr, powerCCstorey=powerCCstorey, powerCCquantile=powerCCquantile,
                      powerCCfisher=powerCCfisher, powerCClinear=powerCClinear,
                      powerKS=powerKS, powerKSPermut=powerKSPermut,
                      powerL2Permut=powerL2Permut, powerOCSVM=powerOCSVM)
    else:
        in_data = np.load(f"{loadname}.npz")
        pi_arr = in_data["pi_arr"]
        power = np.zeros((len(pi_arr), 8), dtype=np.float64)
        power[:, 0] = in_data["powerCCstorey"]
        power[:, 1] = in_data["powerCCquantile"]
        power[:, 2] = in_data["powerCCfisher"]
        power[:, 3] = in_data["powerCClinear"]
        power[:, 4] = in_data["powerKS"]
        power[:, 5] = in_data["powerKSPermut"]
        power[:, 6] = in_data["powerL2Permut"]
        power[:, 7] = in_data["powerOCSVM"]

        fig = plt.figure(figsize=fsize)
        plt.plot(pi_arr, in_data["powerCCstorey"], "-", color="tab:blue", label="Storey")
        plt.plot(pi_arr, in_data["powerCCquantile"], "-", color="tab:pink", label="Quantile")
        plt.plot(pi_arr, in_data["powerCCfisher"], "-", color="tab:red", label="Fisher")
        plt.plot(pi_arr, in_data["powerCClinear"], "-", color="tab:purple", label="Summation")
        plt.plot(pi_arr, in_data["powerKS"], "-", color="tab:orange", label="Asymptotic Kolmogorov-Smirnov")
        plt.plot(pi_arr, in_data["powerKSPermut"], "-", color="tab:green", label="Permutation Kolmogorov-Smirnov")
        plt.plot(pi_arr, in_data["powerL2Permut"], "-", color="tab:brown", label="Permutation L2")
        plt.plot(pi_arr, in_data["powerOCSVM"], "-", color="tab:cyan", label="Permutation OCSVM")
        plt.xlabel(r"$\pi$")
        plt.ylabel("Power")
        plt.legend()
        if save_res is True:
            plt.savefig(f"{loadname}.png", dpi=500, bbox_inches="tight")
        plt.show()

        if save_res is True:
            with open(f"{loadname}.txt", "w") as file:
                name_list = ["Storey", "Quantile", "Fisher", "Linear", "KS", "KSPermut", "L2Permut", "OCSVM"]
                color_list = ["color2", "color3", "color4", "color5", "color6", "color7", "color8", "color9"]
                marker_list = ["square", "diamond", "star", "triangle"]
                linestyle_list = ["solid", "dashed", "dotted", "dashdotted"]
                for idx in range(8):
                    file.write(f"\\addplot[semithick, mark={marker_list[0]}"+", mark options={solid},"+f" {linestyle_list[0]}, {color_list[idx]}]\n")
                    file.write("table{%\n")
                    for x, y in zip(pi_arr, power[:, idx]):
                        file.write(f"{x:.4f}  {y:.4f}\n")
                    # file.write("};\n\\addlegendentry{"+f"{name_list[idx]} ($m =$ " + f"{m:d}" + ")}\n\n")
                    file.write("};\\label{plot:"+f"{name_list[idx]}_"+"}\n")


