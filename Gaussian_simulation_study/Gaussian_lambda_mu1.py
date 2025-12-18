#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:03:51 2025

In this script we do testing of the null hypothesis H_0 : \pi \leq \pi_th
using the test statistic, T = \sum_i 1[p_i > \lambda], for various
values of \lambda and \mu_1.
The data in consideration is Gaussian.
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from rejection_region_storey import Storey_pvalue


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    np.random.seed(42)
    inloop_plotting = False

    # =============================================================================
    # Setup and simulate
    # =============================================================================
    sims = 5000

    mu0 = 0
    sigma0 = 1
    sigma1 = 1

    n_train = 200
    n = 200
    m = 50

    pi = 0.2
    pi_th = 0.2

    alpha = 0.05

    mu1_arr = np.linspace(1, 4, 4)
    lambda_arr = np.linspace(1e-02, 1-1e-02, 6)

    len_mu1 = len(mu1_arr)
    len_lambda = len(lambda_arr)

    # =============================================================================
    # Make p-value look-up table
    # =============================================================================
    p_hat_lookup = np.zeros((len_lambda, m+1), dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
        for j in range(m+1):
            p_hat_lookup[lambda_idx, j] = Storey_pvalue(pi_th, n, m, j, lambda_)

    # =============================================================================
    # Simulate data
    # =============================================================================
    p_marg_all_arr = np.zeros((sims, len_mu1, m), dtype=np.float64)
    m0_arr = np.zeros(sims, dtype=np.int32)
    for sim_idx in range(sims):
        Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
        OCSVM = OneClassSVM()
        OCSVM.fit(Xtrain)
        SC = OCSVM.score_samples(XC).astype(np.float32)
        m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
        m0_arr[sim_idx] = m0
        XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
        for mu1_idx, mu1 in enumerate(mu1_arr):
            XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
            XT = np.concatenate((XT0, XT1), axis=0)
            ST = OCSVM.score_samples(XT).astype(np.float32) # Compute conformal scores
            for j in range(m):
                p_marg_all_arr[sim_idx, mu1_idx, j] = (1 + np.sum(SC <= ST[j]))/(n+1)

    # =============================================================================
    # Estimate rejection rate
    # =============================================================================
    rejectBool = np.zeros((sims, len_lambda, len_mu1), dtype=bool)
    TestStat = np.zeros((sims, len_lambda, len_mu1), dtype=np.int32)
    sum_alternative = np.zeros((sims, len_lambda, len_mu1), dtype=np.float32)
    for sim_idx in range(sims):
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            for mu1_idx, mu1 in enumerate(mu1_arr):
                sum_alternative[sim_idx, lambda_idx, mu1_idx] = np.sum(p_marg_all_arr[sim_idx, mu1_idx, m0_arr[sim_idx]:] > lambda_)/(m-m0_arr[sim_idx])
                test_stat = np.sum(p_marg_all_arr[sim_idx, mu1_idx, :] > lambda_)
                TestStat[sim_idx, lambda_idx, mu1_idx] = test_stat
                if p_hat_lookup[lambda_idx, test_stat] <= alpha:
                    rejectBool[sim_idx, lambda_idx, mu1_idx] = True
                else:
                    rejectBool[sim_idx, lambda_idx, mu1_idx] = False

    rejection_rate = np.sum(rejectBool, axis=0)/sims

    plt.figure(figsize=fsize)
    spec = plt.pcolormesh(mu1_arr, lambda_arr, rejection_rate, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="rejection rate")
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\lambda$")
    plt.show()

    color_list = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for mu1_idx, mu1 in enumerate(mu1_arr):
        plt.figure(figsize=fsize)
        plt.title(f"mu1 = {mu1}")
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            TS_density = np.zeros(m+1)
            for j in range(m+1):
                TS_density[j] = sum(np.where(TestStat[:, lambda_idx, mu1_idx] == j, 1, 0))
            TS_density = TS_density/np.sum(TS_density)
            low = m*(1-pi)*(1-lambda_)
            plt.bar(np.arange(m+1), TS_density, alpha=0.5, label=f"{lambda_:.2f}, {np.std(TestStat[:, lambda_idx, mu1_idx])/np.mean(TestStat[:, lambda_idx, mu1_idx]):.2f}, {np.mean(sum_alternative[:, lambda_idx, mu1_idx]):.2f}", color=color_list[lambda_idx])
            plt.axvline(low, color=color_list[lambda_idx], linestyle="dashed")
            plt.axvline(np.mean(TestStat[:, lambda_idx, mu1_idx]), color=color_list[lambda_idx])
        plt.legend()
        plt.xlabel(r"Test statistic, $T(\lambda)$")
        plt.ylabel("Density")
        plt.show()

    print("Maximum rejection rate: ", np.max(rejection_rate), "  alpha: ", alpha)
