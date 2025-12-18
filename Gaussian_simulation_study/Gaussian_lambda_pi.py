#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:12:51 2025

In this script we do testing of the null hypothesis H_0 : \pi \leq \pi_th
using the test statistic, T = \sum_i 1[p_i > \lambda], for various
values of \pi and \lambda.
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
    sims = 1000

    mu0 = 0
    sigma0 = 1
    mu1 = 4
    sigma1 = 1

    n_train = 200
    n = 400
    m = 100

    pi_th = 0.5

    alpha = 0.05

    # pi_arr = np.linspace(0.1, 0.9, 17)
    pi_arr = np.linspace(0.1, pi_th, 20, endpoint=True)
    # pi_arr = np.linspace(pi_th, 0.9, 20)
    lambda_arr = np.logspace(-2.2, 0, 25, endpoint=False)

    len_pi = len(pi_arr)
    len_lambda = len(lambda_arr)

    # =============================================================================
    # Determine rejection region
    # =============================================================================
    # ub_pi_th_arr = np.zeros(len_lambda, dtype=np.float64)
    # for lambda_idx, lambda_ in enumerate(lambda_arr):
    #     _, ub_pi_th_arr[lambda_idx] = rejection_region_wrapper(pi_th, n, m, lambda_, alpha)

    p_hat_lookup = np.zeros((len_lambda, m+1), dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
        for j in range(m+1):
            p_hat_lookup[lambda_idx, j] = Storey_pvalue(pi_th, n, m, j, lambda_)

    # =============================================================================
    # Simulate data
    # =============================================================================
    p_marg_all_arr = np.zeros((sims, len_pi, m), dtype=np.float64)
    for sim_idx in range(sims):
        Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
        OCSVM = OneClassSVM()
        OCSVM.fit(Xtrain)
        SC = OCSVM.score_samples(XC).astype(np.float32)
        for pi_idx, pi in enumerate(pi_arr):
            m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
            XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
            XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
            XT = np.concatenate((XT0, XT1), axis=0)
            ST = OCSVM.score_samples(XT).astype(np.float32) # Compute conformal scores
            for j in range(m):
                p_marg_all_arr[sim_idx, pi_idx, j] = (1 + np.sum(SC <= ST[j]))/(n+1)

    # =============================================================================
    # Estimate rejection rate
    # =============================================================================
    rejectBool = np.zeros((sims, len_lambda, len_pi), dtype=bool)
    for sim_idx in range(sims):
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            for pi_idx, pi in enumerate(pi_arr):
                test_stat = np.sum(p_marg_all_arr[sim_idx, pi_idx, :] > lambda_)
                if p_hat_lookup[lambda_idx, test_stat] <= alpha:
                    rejectBool[sim_idx, lambda_idx, pi_idx] = True
                else:
                    rejectBool[sim_idx, lambda_idx, pi_idx] = False

    rejection_rate = np.sum(rejectBool, axis=0)/sims

    plt.figure(figsize=fsize)
    spec = plt.pcolormesh(pi_arr, lambda_arr, rejection_rate, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="rejection rate")
    plt.xlabel(r"$\pi$")
    plt.ylabel(r"$\lambda$")
    plt.show()

    print("Maximum rejection rate: ", np.max(rejection_rate), "  alpha: ", alpha)

    # if False:
    #     plt.figure(figsize=fsize)
    #     plt.plot(lambda_arr, ub_pi_th_arr, color="tab:blue")
    #     plt.xlabel(r"$\lambda$")
    #     plt.ylabel("upper bound")
    #     plt.show()

    #     plt.figure(figsize=fsize)
    #     plt.plot(np.repeat(ub_pi_th_arr, len_pi).flatten(), rejection_rate.flatten(), "o", color="tab:blue")
    #     plt.ylabel("rejection rate")
    #     plt.xlabel("upper bound")
    #     plt.show()

    #     mean_rr_lambda = np.zeros(len_lambda, dtype=np.float32)
    #     # max_rr_lambda = np.zeros(len_lambda, dtype=np.float32)
    #     for lambda_idx, _ in enumerate(lambda_arr):
    #         mean_rr_lambda[lambda_idx] = np.mean(rejection_rate.flatten()[np.repeat(ub_pi_th_arr, len_pi).flatten() == ub_pi_th_arr[lambda_idx]])
    #         # max_rr_lambda[lambda_idx] = np.max(rejection_rate.flatten()[np.repeat(ub_pi_th_arr, len_pi).flatten() == ub_pi_th_arr[lambda_idx]])

    #     plt.figure(figsize=fsize)
    #     plt.plot(ub_pi_th_arr, mean_rr_lambda, "o", color="tab:blue", label="mean")
    #     # plt.plot(ub_pi_th_arr, max_rr_lambda, "o", color="tab:red", label="max")
    #     # for lambda_idx, lambda_ in enumerate(lambda_arr):
    #     #     plt.annotate(f"{lambda_:.3f}", (ub_pi_th_arr[lambda_idx], mean_rr_lambda[lambda_idx]))
    #     # for lambda_idx, lambda_ in enumerate(lambda_arr):
    #     #     plt.annotate(f"{lambda_:.3f}", (ub_pi_th_arr[lambda_idx], max_rr_lambda[lambda_idx]))
    #     plt.legend()
    #     plt.ylabel("average rejection rate")
    #     plt.xlabel("upper bound")
    #     plt.show()
