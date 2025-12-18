#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:11:32 2025

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

from rejection_region_storey import Storey_pvalue
from rejection_region_quantile import quantile_pvalue
from rejection_region_asymptotic import Fisher_pvalue
from Benjamini_Hochberg import Benjamini_Hochberg_procedure, compute_FDR, compute_TDR


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    np.random.seed(49)
    inloop_plotting = False

    # =============================================================================
    # Setup and simulate
    # =============================================================================
    sims = 50

    mu0 = 0
    sigma0 = 1
    mu1 = 3
    sigma1 = 1

    n_train = 200
    n = 200

    K = 10
    K0 = 5
    m_iter = 20
    T = 5

    pi_null = 0.5
    pi_alt = 0.7
    pi_th = 0.5
    assert pi_null <= pi_th and pi_alt > pi_th

    qt_arr = np.linspace(0.001, 0.5, 100)/T
    lambda_ = 5 / (n+1)

    rejectBool_storey = np.zeros((sims, len(qt_arr), T, K), dtype=bool)
    rejectBool_quantile = np.zeros((sims, len(qt_arr), T, K), dtype=bool)
    rejectBool_fisher = np.zeros((sims, len(qt_arr), T, K), dtype=bool)
    for i in range(sims):
        Xtrain = np.random.normal(loc=mu0, scale=sigma0, size=(n_train, 2)).astype(np.float32)
        XC = np.random.normal(loc=mu0, scale=sigma0, size=(n, 2)).astype(np.float32)

        OCSVM = OneClassSVM()
        OCSVM.fit(Xtrain)
        SC = OCSVM.score_samples(XC).astype(np.float32)

        p_hat_storey = np.zeros((K, T), dtype=np.float32)
        p_hat_quantile = np.zeros((K, T), dtype=np.float32)
        p_hat_fisher = np.zeros((K, T), dtype=np.float32)
        ST = np.zeros((K, 0), dtype=np.float32)
        pmarg_all = [np.array([]) for i in range(K)]
        m = 0
        for t in range(T):
            m += m_iter
            for k in range(K):
                if k < K0:
                    m0_iter = np.random.binomial(m_iter, p=1-pi_null)
                else:
                    m0_iter = np.random.binomial(m_iter, p=1-pi_alt)
                XT0 = np.random.normal(loc=mu0, scale=sigma0, size=(m0_iter, 2)).astype(np.float32)
                XT1 = np.random.normal(loc=mu1, scale=sigma1, size=(m_iter-m0_iter, 2)).astype(np.float32)
                XT = np.concatenate((XT0, XT1), axis=0)
                # ST[k] = np.hstack((ST[k], OCSVM.score_samples(XT).astype(np.float32)))
                ST = OCSVM.score_samples(XT).astype(np.float32)

                pmarg_all_iter = np.zeros(m_iter, dtype=np.float32)
                for j in range(m_iter):
                    pmarg_all_iter[j] = (1 + np.sum(SC <= ST[j]))/(n+1)
                pmarg_all[k] = np.concatenate((pmarg_all[k], pmarg_all_iter), axis=0)

                i0 = int(m-ssbinom.ppf(0.998, m, pi_th))

                pmarg_k_arr = np.array(pmarg_all[k])
                Tstorey = round(np.sum(pmarg_k_arr > lambda_))
                Tquantile = round(np.sort(pmarg_k_arr)[m-i0-1] * (n+1))
                Tfisher = -2*np.sum(np.log((n+2)/(n+1) - pmarg_k_arr))

                p_hat_storey[k, t] = Storey_pvalue(pi_th, n, m, Tstorey, lambda_)
                p_hat_quantile[k, t] = quantile_pvalue(pi_th, n, m, Tquantile, i0)
                p_hat_fisher[k, t] = Fisher_pvalue(pi_th, n, m, Tfisher)

            K0_hat = K0
            for alpha_idx, alpha in enumerate(qt_arr):
                rejectBool_storey[i, alpha_idx, t] = Benjamini_Hochberg_procedure(p_hat_storey[:, t], alpha, K0_hat, K)
                rejectBool_quantile[i, alpha_idx, t] = Benjamini_Hochberg_procedure(p_hat_quantile[:, t], alpha, K0_hat, K)
                rejectBool_fisher[i, alpha_idx, t] = Benjamini_Hochberg_procedure(p_hat_fisher[:, t], alpha, K0_hat, K)

            for alpha_idx, alpha in enumerate(qt_arr):
                for k in range(K):
                    if rejectBool_storey[i, alpha_idx, t-1, k] == True:
                        rejectBool_storey[i, alpha_idx, t, k] = True
                    if rejectBool_quantile[i, alpha_idx, t-1, k] == True:
                        rejectBool_quantile[i, alpha_idx, t, k] = True
                    if rejectBool_fisher[i, alpha_idx, t-1, k] == True:
                        rejectBool_fisher[i, alpha_idx, t, k] = True

    FDR_storey = np.zeros((len(qt_arr), T), dtype=np.float32)
    FDR_quantile = np.zeros((len(qt_arr), T), dtype=np.float32)
    FDR_fisher = np.zeros((len(qt_arr), T), dtype=np.float32)
    for alpha_idx, alpha in enumerate(qt_arr):
        for t in range(T):
            FDR_storey[alpha_idx, t], _ = compute_FDR(rejectBool_storey[:, alpha_idx, t, :], K0, sims)
            FDR_quantile[alpha_idx, t], _ = compute_FDR(rejectBool_quantile[:, alpha_idx, -t, :], K0, sims)
            FDR_fisher[alpha_idx, t], _ = compute_FDR(rejectBool_fisher[:, alpha_idx, t, :], K0, sims)

    TDR_storey = np.zeros((len(qt_arr), T), dtype=np.float32)
    TDR_quantile = np.zeros((len(qt_arr), T), dtype=np.float32)
    TDR_fisher = np.zeros((len(qt_arr), T), dtype=np.float32)
    for alpha_idx, alpha in enumerate(qt_arr):
        for t in range(T):
            TDR_storey[alpha_idx, t], _ = compute_TDR(rejectBool_storey[:, alpha_idx, t, :], K0, K, sims)
            TDR_quantile[alpha_idx, t], _ = compute_TDR(rejectBool_quantile[:, alpha_idx, t, :], K0, K, sims)
            TDR_fisher[alpha_idx, t], _ = compute_TDR(rejectBool_fisher[:, alpha_idx, t, :], K0, K, sims)

    # print(f"False discovery rate) Storey: {FDR_storey:.4f}    Quantile: {FDR_quantile:.4f}    Fisher: {FDR_fisher:.4f}")
    # print(f" True discovery rate) Storey: {TDR_storey:.4f}    Quantile: {TDR_quantile:.4f}    Fisher: {TDR_fisher:.4f}")

    fig = plt.figure(figsize=fsize)
    plt.plot(qt_arr, FDR_storey[:, -1], color="tab:orange", label="Storey")
    plt.plot(qt_arr, FDR_quantile[:, -1], color="tab:purple", label="quantile")
    plt.plot(qt_arr, FDR_fisher[:, -1], color="tab:green", label="Fisher")
    plt.plot(qt_arr, qt_arr*T, color="k")
    plt.xlabel(r"$q^t$")
    plt.ylabel("FDR")
    plt.legend()
    # plt.savefig(, bbox_inches="tight", dpi=500)
    plt.show()

    fig = plt.figure(figsize=fsize)
    plt.plot(qt_arr, TDR_storey[:, -1], color="tab:orange", label="Storey")
    plt.plot(qt_arr, TDR_quantile[:, -1], color="tab:purple", label="quantile")
    plt.plot(qt_arr, TDR_fisher[:, -1], color="tab:green", label="Fisher")
    plt.xlabel(r"$q^t$")
    plt.ylabel("TDR")
    plt.legend()
    # plt.savefig(, bbox_inches="tight", dpi=500)
    plt.show()

    fig = plt.figure(figsize=fsize)
    plt.plot(np.arange(T)*m_iter, TDR_storey[10, :], color="tab:orange", label="Storey")
    plt.plot(np.arange(T)*m_iter, TDR_quantile[10, :], color="tab:purple", label="quantile")
    plt.plot(np.arange(T)*m_iter, TDR_fisher[10, :], color="tab:green", label="Fisher")
    plt.xlabel("t")
    plt.ylabel("TDR")
    plt.legend()
    # plt.savefig(, bbox_inches="tight", dpi=500)
    plt.show()