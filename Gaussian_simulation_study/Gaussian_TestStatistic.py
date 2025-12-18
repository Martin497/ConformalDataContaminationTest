#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:50:37 2025

In this script we do testing of the null hypothesis H_0 : \pi \leq \pi_th
using the proposed test statistics.
The data in consideration is Gaussian.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from scipy.stats import binom as ssbinom

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
    sims = 500

    mu0 = 0
    sigma0 = 1
    mu1 = 4
    sigma1 = 1

    n_train = 200
    n = 500
    m = 200

    pi = 0.6
    pi_th = 0.5

    alpha = 0.05
    lambda_ = n//2 / (n+1)
    # i0 = m-m//2
    i0 = int(m-ssbinom.ppf(0.999, m, pi_th)) # binom.ppf(0.0001, m, 1-pi)
    B = 1000

    test_handler = ConformalContaminationTest()

    rejectBool_storey = np.zeros(sims, dtype=bool)
    rejectBool_quantile = np.zeros(sims, dtype=bool)
    rejectBool_fisher = np.zeros(sims, dtype=bool)
    rejectBool_linear = np.zeros(sims, dtype=bool)
    rejectBool_shiftFisher = np.zeros(sims, dtype=bool)
    for i in range(sims):
        Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)
        m0 = np.random.binomial(m, p=1-pi) # (1-\pi) % of test data is from null
        XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
        XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m-m0, 2)).astype(np.float32)
        XT = np.concatenate((XT0, XT1), axis=0)

        if inloop_plotting is True:
            plt.figure(figsize=fsize)
            plt.scatter(XC[:, 0], XC[:, 1], color="tab:blue", label="Null")
            plt.scatter(XT[:, 0], XT[:, 1], color="tab:red", label="Test")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

        # =============================================================================
        # Compute conformal scores
        # =============================================================================
        OCSVM = OneClassSVM()
        OCSVM.fit(Xtrain)
        SC = OCSVM.score_samples(XC).astype(np.float32)
        ST = OCSVM.score_samples(XT).astype(np.float32)

        if inloop_plotting is True:
            plt.figure(figsize=fsize)
            plt.hist(SC, bins=20, density=True, alpha=0.7, color="tab:blue", label="Null")
            plt.hist(ST, bins=20, density=True, alpha=0.7, color="tab:red", label="Test")
            plt.xlabel("Conformity score")
            plt.legend()
            plt.show()

        # =============================================================================
        # Rejection based on \hat{\pi}
        # =============================================================================
        pstorey, pquantile, pfisher, pstorey_bootstrap, pquantile_bootstrap, plinear, pshiftFisher \
            = test_handler.all_conformal_contamination_tests(SC, ST, pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0, B=B)

        if pstorey <= alpha:
            rejectBool_storey[i] = True

        if pquantile <= alpha:
            rejectBool_quantile[i] = True

        if pfisher <= alpha:
            rejectBool_fisher[i] = True

        if plinear <= alpha:
            rejectBool_linear[i] = True

        if pshiftFisher <= alpha:
            rejectBool_shiftFisher[i] = True

    print("Empirical rejection proportion Storey:          ", np.sum(rejectBool_storey)/sims)
    print("Empirical rejection proportion Quantile:        ", np.sum(rejectBool_quantile)/sims)
    print("Empirical rejection proportion Fisher:          ", np.sum(rejectBool_fisher)/sims)
    print("Empirical rejection proportion linear:          ", np.sum(rejectBool_linear)/sims)
    print("Empirical rejection proportion shifted Fisher:  ", np.sum(rejectBool_shiftFisher)/sims)

