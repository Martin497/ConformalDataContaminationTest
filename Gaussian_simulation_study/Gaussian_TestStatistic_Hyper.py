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

# from math import factorial
# from sklearn.svm import OneClassSVM
from scipy.stats import binom
from scipy.stats import nhypergeom

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

# from rejection_region_storey import Storey_pvalue
# from rejection_region_quantile import quantile_pvalue
# from rejection_region_asymptotic import Fisher_pvalue
from ConformalContaminationTestModule import ConformalContaminationTest


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    np.random.seed(42)
    inloop_plotting = False

    # =============================================================================
    # Setup and simulate
    # =============================================================================
    sims = 10000

    mu0 = 0
    sigma0 = 1
    sigma1 = 1

    # n_train = 200
    n = 200
    m = 50

    pi = 0.7
    pi_th = 0.5

    alpha = 0.05

    mu1_arr = np.linspace(1, 5, 5)
    lambda_arr = np.arange(0, n+2, 1) / (n+1)
    # i0_arr = np.arange(0, m//2, 1)
    i0_arr = np.arange(0, m-1, 1)
    B = 1

    len_mu1 = len(mu1_arr)
    len_lambda = len(lambda_arr)
    len_i0 = len(i0_arr)

    test_handler = ConformalContaminationTest()

    # =============================================================================
    # Make p-value look-up tables
    # =============================================================================
    p_hat_storey_lookup = np.zeros((len_lambda, m+1), dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
        for j in range(m+1):
            # p_hat_storey_lookup[lambda_idx, j] = Storey_pvalue(pi_th, n, m, j, lambda_)
            p_hat_storey_lookup[lambda_idx, j] = test_handler.Storey_pvalue(j, pi_th=pi_th, n=n, m=m, lambda_=lambda_)
    p_hat_quantile_lookup = np.zeros((len_i0, n+2), dtype=np.float64)
    for i0_idx, i0 in enumerate(i0_arr):
        for j in range(n+2):
            # p_hat_quantile_lookup[i0_idx, j] = quantile_pvalue(pi_th, n, m, j, i0)
            p_hat_quantile_lookup[i0_idx, j] = test_handler.quantile_pvalue(j, pi_th=pi_th, n=n, m=m, i0=i0)

    # =============================================================================
    # Simulate data
    # =============================================================================
    # calibration_scores = np.zeros((sims, n))
    # test_scores = np.zeros((sims, len_mu1, m))
    p_marg_all_arr = np.zeros((sims, len_mu1, m), dtype=np.float64)
    m0_arr = np.zeros(sims, dtype=np.int32)
    for sim_idx in range(sims):
        # Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
        # OCSVM = OneClassSVM()
        # OCSVM.fit(Xtrain)
        # SC = OCSVM.score_samples(XC).astype(np.float32)
        SC = -np.linalg.norm(XC, axis=-1).astype(np.float32)

        # calibration_scores[sim_idx] = SC
        m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
        m0_arr[sim_idx] = m0
        XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
        for mu1_idx, mu1 in enumerate(mu1_arr):
            XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
            XT = np.concatenate((XT0, XT1), axis=0)
            # ST = OCSVM.score_samples(XT).astype(np.float32) # Compute conformal scores
            ST = -np.linalg.norm(XT, axis=-1).astype(np.float32)
            # test_scores[sim_idx, mu1_idx] = ST
            # for j in range(m):
            #     p_marg_all_arr[sim_idx, mu1_idx, j] = (1 + np.sum(SC <= ST[j]))/(n+1)
            p_marg_all_arr[sim_idx, mu1_idx] = test_handler.compute_conformal_pvalues(SC, ST)

    # =============================================================================
    # Estimate rejection rate
    # =============================================================================
    rejectBool_storey = np.zeros((sims, len_lambda, len_mu1), dtype=bool)
    rejectBool_quantile = np.zeros((sims, len_i0, len_mu1), dtype=bool)
    rejectBool_fisher = np.zeros((sims, len_mu1), dtype=bool)
    rejectBool_linear = np.zeros((sims, len_mu1), dtype=bool)
    TestStat_storey = np.zeros((sims, len_lambda, len_mu1), dtype=np.int32)
    TestStat_quantile = np.zeros((sims, len_i0, len_mu1), dtype=np.int32)
    TestStat_fisher = np.zeros((sims, len_i0, len_mu1), dtype=np.int32)
    # sum_alternative = np.zeros((sims, len_lambda, len_mu1), dtype=np.float32)
    for sim_idx in range(sims):
        # if sim_idx % 10 == 0:
        #     print("Simulation number:", sim_idx)
        ### Storey loop ###
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            for mu1_idx, mu1 in enumerate(mu1_arr):
                # if m-m0_arr[sim_idx] == 0:
                #     sum_alternative[sim_idx, lambda_idx, mu1_idx] = 0
                # else:
                #     sum_alternative[sim_idx, lambda_idx, mu1_idx] = np.sum(p_marg_all_arr[sim_idx, mu1_idx, m0_arr[sim_idx]:] > lambda_)/(m-m0_arr[sim_idx])
                test_stat = test_handler.Storey_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], lambda_=lambda_)
                # test_stat = np.sum(p_marg_all_arr[sim_idx, mu1_idx, :] > lambda_)
                TestStat_storey[sim_idx, lambda_idx, mu1_idx] = test_stat
                # pvalue = test_handler.Storey_pvalue(test_stat, pi_th=pi_th, n=n, m=m, lambda_=lambda_)
                # if pvalue <= alpha:
                #     rejectBool_storey[sim_idx, lambda_idx, mu1_idx] = True
                if p_hat_storey_lookup[lambda_idx, test_stat] <= alpha:
                    rejectBool_storey[sim_idx, lambda_idx, mu1_idx] = True
                # else:
                #     rejectBool_storey[sim_idx, lambda_idx, mu1_idx] = False
        ### Quantile loop ###
        for i0_idx, i0 in enumerate(i0_arr):
            for mu1_idx, mu1 in enumerate(mu1_arr):
                test_stat = test_handler.quantile_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], n=n, i0=i0)
                # test_stat = round(np.sort(p_marg_all_arr[sim_idx, mu1_idx, :])[m-i0-1] * (n+1))
                TestStat_quantile[sim_idx, i0_idx, mu1_idx] = test_stat
                # pvalue = test_handler.quantile_pvalue(test_stat, pi_th=pi_th, n=n, m=m, i0=i0)
                # if pvalue <= alpha:
                #     rejectBool_quantile[sim_idx, i0_idx, mu1_idx] = True
                if p_hat_quantile_lookup[i0_idx, test_stat] <= alpha:
                    rejectBool_quantile[sim_idx, i0_idx, mu1_idx] = True
                # else:
                #     rejectBool_quantile[sim_idx, i0_idx, mu1_idx] = False
        ### Fisher ###
        for mu1_idx, mu1 in enumerate(mu1_arr):
            test_stat = test_handler.shifted_Fisher_test_statistic(p_marg_all_arr[sim_idx, mu1_idx], n=n)
            # test_stat = -2*np.sum(np.log((n+2)/(n+1) - p_marg_all_arr[sim_idx, mu1_idx, :]))
            TestStat_fisher[sim_idx, mu1_idx] = test_stat
            pvalue = test_handler.shifted_Fisher_pvalue(test_stat, pi_th=pi_th, n=n, m=m)
            if pvalue <= alpha:
                rejectBool_fisher[sim_idx, mu1_idx] = True
            # if Fisher_pvalue(pi_th, n, m, test_stat) <= alpha:
            #     rejectBool_fisher[sim_idx, mu1_idx] = True
            # else:
            #     rejectBool_fisher[sim_idx, mu1_idx] = False
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

    # print("Maximum rejection rate Storey:   ", np.max(rejection_rate_storey), "  alpha: ", alpha)
    # print("Maximum rejection rate Quantile: ", np.max(rejection_rate_quantile), "  alpha: ", alpha)

    # =============================================================================
    # Storey color plot
    # =============================================================================
    plt.figure(figsize=fsize)
    spec = plt.pcolormesh(mu1_arr, lambda_arr, rejection_rate_storey, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="rejection rate")
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\lambda$")
    # plt.savefig(f"TestStatistic_Hyper/storey_color_n{n:d}_m{m:}_pi{pi:.1f}_pith{pi_th:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
    plt.show()

    # =============================================================================
    # Storey bar plots
    # =============================================================================
    color_list = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for mu1_idx, mu1 in enumerate(mu1_arr):
        plt.figure(figsize=fsize)
        plt.title(f"mu1 = {mu1}")
        iter_ = 0
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            if lambda_ == 1/(n+1) or lambda_ == (n-1)/(n+1) or lambda_ == n//3 / (n+1) or lambda_ == (2*n)//3 / (n+1):
                iter_ += 1
                TS_density = np.zeros(m+1)
                for j in range(m+1):
                    TS_density[j] = sum(np.where(TestStat_storey[:, lambda_idx, mu1_idx] == j, 1, 0))
                TS_density = TS_density/np.sum(TS_density)
                low = m*(1-pi)*(1-lambda_)
                # high = m*(1-lambda_)
                # dev = np.std(TestStat_storey[:, lambda_idx, mu1_idx])/np.mean(TestStat_storey[:, lambda_idx, mu1_idx])
                # PH1 = np.mean(sum_alternative[:, lambda_idx, mu1_idx])
                plt.bar(np.arange(m+1), TS_density, alpha=0.5, label=f"$\lambda$={round((n+1)*lambda_):d}/(n+1)", color=color_list[iter_])
                plt.axvline(low, color=color_list[iter_], linestyle="dashed")
                # plt.axvline(high, color=color_list[iter_], linestyle="dotted")
                plt.axvline(np.mean(TestStat_storey[:, lambda_idx, mu1_idx]), color=color_list[iter_])
        plt.legend()
        plt.xlabel(r"Test statistic, $T_{storey}(\lambda)$")
        plt.ylabel("Density")
        # plt.savefig(f"TestStatistic_Hyper/storey_bar_mu{int(mu1):d}_n{n:d}_m{m:}_pi{pi:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
        plt.show()

    # =============================================================================
    # Storey functional boxplot
    # =============================================================================
    mu1_idx = 1
    markersize = 1.5

    # fig = plt.figure(figsize=fsize)
    # ax = fig.add_subplot(111)
    # plt.plot(lambda_arr, np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.50, axis=0), "o", color="navy", label="Empirical median", markersize=markersize)
    # ax.fill_between(lambda_arr, np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.05, axis=0), np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.25, axis=0), color="cornflowerblue")
    # ax.fill_between(lambda_arr, np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.25, axis=0), np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.50, axis=0), color="tab:blue")
    # ax.fill_between(lambda_arr, np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.50, axis=0), np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.75, axis=0), color="tab:blue")
    # ax.fill_between(lambda_arr, np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.75, axis=0), np.nanquantile(TestStat_storey[:, :, mu1_idx], q=0.95, axis=0), color="cornflowerblue")
    # plt.plot(lambda_arr, m*(1-pi)*(1-lambda_arr), "o", color="tab:red", label="Asymptotic approximation", markersize=markersize)
    # plt.legend()
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel(r"Test statistic, $T_{storey}(\lambda)$")
    # plt.show()

    mean = np.mean(TestStat_storey[:, :, mu1_idx], axis=0)[:-1]
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111)
    plt.plot(lambda_arr[:-1], np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.50, axis=0)/mean, "o", color="navy", label="Normalized empirical median", markersize=markersize)
    ax.fill_between(lambda_arr[:-1], np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.05, axis=0)/mean, np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.25, axis=0)/mean, color="cornflowerblue")
    ax.fill_between(lambda_arr[:-1], np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.25, axis=0)/mean, np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.50, axis=0)/mean, color="tab:blue")
    ax.fill_between(lambda_arr[:-1], np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.50, axis=0)/mean, np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.75, axis=0)/mean, color="tab:blue")
    ax.fill_between(lambda_arr[:-1], np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.75, axis=0)/mean, np.nanquantile(TestStat_storey[:, :-1, mu1_idx], q=0.95, axis=0)/mean, color="cornflowerblue")
    plt.plot(lambda_arr[:-1], m*(1-pi)*(1-lambda_arr[:-1])/mean, "o", color="tab:red", label="Normalized asymptotic approximation", markersize=markersize)
    plt.plot(lambda_arr[:-1], np.ones(len_lambda-1), color="k", markersize=markersize, linestyle="dashed")
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$T_{storey}(\lambda)/E[T_{storey}(\lambda)]$")
    # plt.savefig(f"TestStatistic_Hyper/storey_funcbox_mu{int(mu1_arr[mu1_idx]):d}_n{n:d}_m{m:}_pi{pi:.1f}.png", bbox_inches="tight", dpi=500)
    plt.show()

    # =============================================================================
    # Storey bias etc... plots
    # =============================================================================
    mu1_idx = 1
    markersize = 3

    # bias = (np.mean(TestStat_storey[:, :-1, mu1_idx], axis=0) - m*(1-pi)*(1-lambda_arr[:-1])) / np.mean(TestStat_storey[:, :-1, mu1_idx], axis=0)
    # dev = np.std(TestStat_storey[:, :-1, mu1_idx], axis=0)/np.mean(TestStat_storey[:, :-1, mu1_idx], axis=0)
    # mse = np.sqrt(np.mean((TestStat_storey[:, :-1, mu1_idx] - m*(1-pi)*(1-lambda_arr[:-1]))**2, axis=0) / np.mean(TestStat_storey[:, :-1, mu1_idx]**2, axis=0))
    # plt.figure(figsize=fsize)
    # plt.plot(lambda_arr[:-1], mse, "o", color="tab:green", label="Normalized MSE", markersize=markersize)
    # plt.plot(lambda_arr[:-1], bias, "o", color="tab:purple", label="Normalized bias", markersize=markersize)
    # plt.plot(lambda_arr[:-1], dev, "o", color="tab:orange", label="Coefficient of variation", markersize=markersize)
    # plt.legend()
    # plt.xlabel(r"$\lambda$")
    # plt.show()

    mean = m*(1-pi)*(1-lambda_arr[:-1])# np.mean(TestStat_storey[:, :-1, mu1_idx], axis=0)
    bias = np.mean(TestStat_storey[:, :-1, mu1_idx], axis=0) - m*(1-pi)*(1-lambda_arr[:-1])
    std = np.std(TestStat_storey[:, :-1, mu1_idx], axis=0)
    # plt.figure(figsize=fsize)
    # plt.plot(lambda_arr[:-1], (bias**2 + std**2), "o", color="tab:green", label="MSE", markersize=markersize)
    # plt.plot(lambda_arr[:-1], bias**2, "o", color="tab:purple", label="Squared bias", markersize=markersize)
    # plt.plot(lambda_arr[:-1], std**2, "o", color="tab:orange", label="Variance", markersize=markersize)
    # plt.legend()
    # plt.xlabel(r"$\lambda$")
    # plt.show()

    plt.figure(figsize=fsize)
    plt.plot(lambda_arr[:-1], (bias**2 + std**2)/mean**2, "o", color="tab:green", label="Normalized MSE", markersize=markersize)
    plt.plot(lambda_arr[:-1], bias**2/mean**2, "o", color="tab:purple", label="Normalized squared bias", markersize=markersize)
    plt.plot(lambda_arr[:-1], std**2/mean**2, "o", color="tab:orange", label="Normalized variance", markersize=markersize)
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"$\lambda$")
    # plt.savefig(f"TestStatistic_Hyper/storey_bias_variance_mu{int(mu1_arr[mu1_idx]):d}.png", bbox_inches="tight", dpi=500)
    plt.show()

    # =============================================================================
    # Quantile color plot
    # =============================================================================
    plt.figure(figsize=fsize)
    spec = plt.pcolormesh(mu1_arr, i0_arr, rejection_rate_quantile, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="rejection rate")
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$i_0$")
    # plt.savefig(f"TestStatistic_Hyper/quantile_color_n{n:d}_m{m:}_pi{pi:.1f}_pith{pi_th:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
    plt.show()

    good_i0 = int(m-binom.ppf(0.9999, m, pi_th)) # binom.ppf(0.0001, m, 1-pi_th)

    # =============================================================================
    # Quantile bar plots
    # =============================================================================

    for mu1_idx, mu1 in enumerate(mu1_arr):
        plt.figure(figsize=fsize)
        plt.title(f"mu1 = {mu1}")
        iter_ = 0
        for i0_idx, i0 in enumerate(i0_arr):
            if i0 == 0 or i0 == m//4 or i0 == m//2.5:
                iter_ += 1
                TS_density = np.zeros(n+1)
                for j in range(1, n+2):
                    TS_density[j-1] = sum(np.where(TestStat_quantile[:, i0_idx, mu1_idx] == j, 1, 0))
                TS_density = TS_density/np.sum(TS_density)
                marginal = nhypergeom.pmf(np.arange(0,n+1,1), n+m*(1-pi), n, m*(1-pi)-i0)
                # low = (m*(1-pi)-i0)*n/(m*(1-pi)+1) + 1
                low = nhypergeom.stats(n+m*(1-pi), n, m*(1-pi)-i0, moments="m") + 1
                # high = (m-i0)*n/(m+1) + 1
                plt.bar(np.arange(1, n+2), TS_density, alpha=0.5, label=f"$i_0$={i0:d}", color=color_list[iter_], width=0.4)
                plt.axvline(low, color=color_list[iter_], linestyle="dashed")
                # plt.axvline(high, color=color_list[iter_], linestyle="dotted")
                plt.axvline(np.mean(TestStat_quantile[:, i0_idx, mu1_idx]), color=color_list[iter_])
                plt.plot(np.arange(1,n+2,1), marginal, color=color_list[iter_], marker="o", linestyle="dashed", markersize=markersize)
        plt.legend()
        plt.xlabel(r"Test statistic, $T_{quantile}(i_0)$")
        plt.ylabel("Density")
        # plt.savefig(f"TestStatistic_Hyper/quantile_bar_mu{int(mu1):d}_n{n:d}_m{m:}_pi{pi:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
        plt.show()

    # =============================================================================
    # Quantile bias etc... plots
    # =============================================================================
    mu1_idx = 0
    mean = ((m-i0_arr)*n/(m+1) + 1)/(n+1)
    bias = np.mean(TestStat_quantile[:, :, mu1_idx]/(n+1), axis=0) - mean
    std = np.std(TestStat_quantile[:, :, mu1_idx]/(n+1), axis=0)
    plt.figure(figsize=fsize)
    plt.plot(i0_arr[:], (bias**2 + std**2), "o", color="tab:green", label="MSE", markersize=markersize)
    plt.plot(i0_arr[:], bias**2, "o", color="tab:purple", label="Squared bias", markersize=markersize)
    plt.plot(i0_arr[:], std**2, "o", color="tab:orange", label="Variance", markersize=markersize)
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"i_0$")
    plt.show()

    # =============================================================================
    # Comparison of test statistics
    # =============================================================================
    plt.figure(figsize=fsize)
    plt.plot(mu1_arr, rejection_rate_fisher, "o", color="tab:green", label="Fisher")
    plt.plot(mu1_arr, np.max(rejection_rate_storey, axis=0), "d", color="tab:orange", label="Storey (opt)")
    plt.plot(mu1_arr, rejection_rate_storey[n//2], "o", color="tab:orange", label="Storey (prac)")
    plt.plot(mu1_arr, np.max(rejection_rate_quantile, axis=0), "x", color="tab:purple", label="quantile (opt)")
    plt.plot(mu1_arr, rejection_rate_quantile[good_i0], "o", color="tab:purple", label="quantile (prac)")
    plt.xlabel(r"$\mu$")
    plt.ylabel("Rejection rate")
    plt.legend()
    plt.show()

    mu1_idx = 1

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
    plt.savefig(f"TestStatistic_Hyper/comparison_mu{int(mu1_arr[mu1_idx]):d}_n{n:d}_m{m:}_pi{pi:.1f}_pith{pi_th:.1f}_alpha{alpha:.2f}.png", bbox_inches="tight", dpi=500)
    plt.show()

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
