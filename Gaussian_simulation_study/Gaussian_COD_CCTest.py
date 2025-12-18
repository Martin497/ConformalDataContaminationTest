#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:20:42 2025

@author: martin
"""

import numpy as np
import matplotlib.pyplot as plt

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from ConformalContaminationTestModule import ConformalContaminationTest
from Benjamini_Hochberg import Benjamini_Hochberg_procedure, Storeys_correction



if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    np.random.seed(42)

    sims = 500

    mu0 = 0
    sigma0 = 1
    mu1 = 4
    sigma1 = 1

    # n_train = 200
    n = 30
    m = 300

    pi = 0.5
    pi_th = 0.1

    alpha = 0.05
    beta = 0.05
    zeta = 0.25

    i0_param = 1.1
    i0 = int(m // i0_param)
    lambda_param = 32
    lambda_ = n//lambda_param / (n+1)

    test_handler = ConformalContaminationTest()

    rejectBool_storey = np.zeros(sims, dtype=bool)
    rejectBool_quantile = np.zeros(sims, dtype=bool)
    rejectBool_fisher = np.zeros(sims, dtype=bool)
    rejectBool_linear = np.zeros(sims, dtype=bool)
    rejectBoolCOD = np.zeros((sims, m), dtype=bool)
    for i in range(sims):
        # Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
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
            = test_handler.almost_all_conformal_contamination_tests(SC, ST, pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0)

        if pstorey <= alpha:
            rejectBool_storey[i] = True
        if pquantile <= alpha:
            rejectBool_quantile[i] = True
        if pfisher <= alpha:
            rejectBool_fisher[i] = True
        if plinear <= alpha:
            rejectBool_linear[i] = True

        conformal_pvalues = test_handler.compute_conformal_pvalues(SC, ST)
        _, m0_hat = Storeys_correction(zeta, conformal_pvalues, m)
        rejectBoolCOD[i] = Benjamini_Hochberg_procedure(conformal_pvalues, beta, m0_hat, m)
