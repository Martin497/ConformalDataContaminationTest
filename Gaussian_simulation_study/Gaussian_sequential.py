#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:53:56 2025

In this script we do sequential testing of the null hypothesis
H_0 : \pi \leq \pi_th using the test statistic, T = \sum_i 1[p_i > \lambda].
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
    np.random.seed(43)

    # =============================================================================
    # Setup and simulate
    # =============================================================================
    sims = 2000
    t_arr = np.arange(10)

    mu0 = 0
    sigma0 = 1
    mu1 = 10
    sigma1 = 1

    n_train_0 = 200
    n_train_iter = 0
    n_0 = 500
    n_iter = 0
    m_0 = 0
    m_iter = 20

    pi = 0.5
    pi_th = 0.5

    alpha = 0.05
    lambda_ = 0.5

    # p_hat_lookup = np.zeros(m+1, dtype=np.float64)
    # for j in range(m+1):
    #     p_hat_lookup[j] = upper_bound_parallelised(pi_th, n, m, j, lambda_)

    rejectBool = np.ones((sims, len(t_arr)), dtype=bool)
    rejectBoolGlobal = np.ones(sims, dtype=bool)
    # rejectBoolEprocess = np.ones((sims, len(t_arr)), dtype=bool)
    # rejectBoolOS = np.zeros(sims, dtype=bool)
    pi_hat_Storey = np.zeros(sims, dtype=np.float32)
    for i in range(sims):
        Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train_0, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n_0, 2)).astype(np.float32)
        m0 = np.random.binomial(m_0, p=1-pi) # (1-\pi) % of test data is from null
        XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
        XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m_0-m0, 2)).astype(np.float32)
        XT = np.concatenate((XT0, XT1), axis=0)
        n = n_0
        m = m_0
        # e_process = 0
        # q_star = alpha
        for t_idx, t in enumerate(t_arr):
            Xtrain_new = np.random.normal(loc=mu0, scale=sigma0, size=(n_train_iter, 2)).astype(np.float32)
            Xtrain = np.concatenate((Xtrain, Xtrain_new), axis=0)
            XC_new = np.random.normal(loc=mu0, scale=sigma0, size=(n_iter, 2)).astype(np.float32)
            XC = np.concatenate((XC, XC_new), axis=0)
            m0 = np.random.binomial(m_iter, p=1-pi) # (1-\pi) % of test data is from null
            XT0_new = np.random.normal(loc=mu0, scale=sigma0, size=(m0, 2)).astype(np.float32)
            XT1_new = np.random.normal(loc=mu1, scale=sigma1, size=(m_iter-m0, 2)).astype(np.float32)
            XT_new = np.concatenate((XT0_new, XT1_new), axis=0)
            XT = np.concatenate((XT, XT_new), axis=0)
            m += m_iter
            n += n_iter

            # =============================================================================
            # Compute conformal scores
            # =============================================================================
            OCSVM = OneClassSVM()
            OCSVM.fit(Xtrain)
            SC = OCSVM.score_samples(XC).astype(np.float32)
            ST = OCSVM.score_samples(XT).astype(np.float32)

            # =============================================================================
            # Rejection based on \hat{\pi}
            # =============================================================================
            pmarg_all = np.zeros(m, dtype=np.float32)
            for j in range(m):
                pmarg_all[j] = (1 + np.sum(SC <= ST[j]))/(n+1)
            test_stat = np.sum(pmarg_all > lambda_)
            p_hat = Storey_pvalue(pi_th, n, m, test_stat, lambda_)

            # if p_hat <= q_star:
            #     rejectBool[i, t_idx] = True
            #     break
            # else:
            #     rejectBool[i, t_idx] = False
            #     q_star = (1-p_hat)

            # e_hat = -np.log2(p_hat)
            # e_hat = (1 - p_hat + p_hat * np.log(p_hat))/(p_hat*(-np.log(p_hat))**2)
            # e_process += 1/len(t_arr) * e_hat
            # if e_process >= 1/alpha:
            #     rejectBoolEprocess[i, t_idx] = True
            #     break
            # else:
            #     rejectBoolEprocess[i, t_idx] = False

            # if p_hat <= alpha:
            # # if p_hat <= 1 - (1 - alpha)**(1/len(t_arr)):
            #     rejectBoolOS[i] = True

            if p_hat <= alpha/len(t_arr):
            # if p_hat <= 1 - (1 - alpha)**(1/len(t_arr)):
                rejectBool[i, t_idx] = True
                break
            else:
                rejectBool[i, t_idx] = False
        rejectBoolGlobal[i] = np.any(rejectBool[i])

    print("Empirical rejection proportion with Bonferroni: ", np.sum(rejectBool, axis=0)/sims,
          " Global: ", np.sum(rejectBoolGlobal)/sims)
    # print("Empirical rejection proportion with e-process:  ", np.sum(rejectBoolEprocess, axis=0)/sims)
    # print("Empirical rejection proportion with stopping:   ", np.sum(rejectBoolOS)/sims)