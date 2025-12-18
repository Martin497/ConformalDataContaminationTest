#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:02:23 2025

This module contains basic functionality for running the Benjamini-Hochberg
procedure, computing false discovery rate, and true discovery rate.
"""

import numpy as np
from math import ceil

def Benjamini_Hochberg_procedure(pvals, alpha, m0, m):
    """
    Basic functionality to make a boolean array for rejection/not rejection
    based on p-values, p_hat, and a test sequence, test_sequence.

    Inputs:
    -------
        pvals : ndarray, size=(m)
            The p-values.
        alpha : float in (0, 1)
            The significance level.
        m0 : integer
            The (estimated) number of true nulls.
        m : integer
            The number of tests.

    Output:
    -------
        rejectBool : ndarray, size=(m,)
            If true reject.
    """
    test_sequence = np.arange(1, m+1)/m0 * alpha
    rejectBool = np.ones(m, dtype=bool)
    sort_ = np.argsort(pvals).astype(np.int16)
    pvals_sorted = pvals[sort_]
    iter_ = m-1
    while iter_ >= 0:
        if pvals_sorted[iter_] <= test_sequence[iter_]:
            rejectBool[sort_[iter_]] = True
            iter_ -= 1
            break
        else:
            rejectBool[sort_[iter_]] = False
            iter_ -= 1
    return rejectBool

def Storeys_correction(lambda_, pvals, m, cap=False):
    """
    Estimating the proportion of true nulls, pi0, using Storey's estimate.

    Inputs:
    -------
        lambda_ : float
            Storey's hyperparameter.
        pvals : ndarray, size=(m)
            The p-values.
        m : integer
            The number of tests.

    Output:
    -------
        pi0_hat : float
            The estimated proportion of true nulls.
        m0_hat : int
            The estimated number of true nulls.
    """
    if cap is True:
        pi0_hat = min(1, (1 + np.sum(pvals > lambda_)) / (m * (1 - lambda_)))
        m0_hat = min(m, ceil(pi0_hat * m))
    else:
        pi0_hat = (1 + np.sum(pvals > lambda_)) / (m * (1 - lambda_))
        m0_hat = ceil(pi0_hat * m)
    return pi0_hat, m0_hat

def Storey_FDR(pvals, lambda_, gamma):
    """
    The direct approach to the FDR of Storey (2002).

    Inputs:
    -------
        pvals : ndarray, size=(m)
            The p-values.
        lambda_ : float in [0, 1)
            Storey's hyperparameter.
        gamma : float in [0, lambda_)
            The rejection region is [0, gamma].
    """
    if gamma <= lambda_:
        m = len(pvals)
        test_stat = np.sum(pvals > lambda_)
        pi0_hat_Storey = test_stat / (m * (1 - lambda_))
        Pr_hat_Storey = max(np.sum(pvals <= gamma), 1) / m
        FDR_hat_Storey = pi0_hat_Storey * gamma / Pr_hat_Storey
    elif gamma > lambda_:
        FDR_hat_Storey = 1
    return min(FDR_hat_Storey, 1)

def Storey_FDR_CV(pvals, gamma, B):
    """
    """
    m = len(pvals)
    # lambda_arr = np.arange(1, n+1, 1)/(n+1)
    lambda_arr = np.linspace(0.05, 0.95, 19)
    len_lambda = len(lambda_arr)

    FDR_hat_Storey = np.zeros(len_lambda, dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
        if gamma <= lambda_:
            test_stat = np.sum(pvals > lambda_)
            pi0_hat_Storey = test_stat / (m * (1 - lambda_))
            Pr_hat_Storey = max(np.sum(pvals <= gamma), 1) / m
            FDR_hat_Storey[lambda_idx] = pi0_hat_Storey * gamma / Pr_hat_Storey
        elif gamma > lambda_:
            FDR_hat_Storey[lambda_idx] = 1
    FDR_hat_Storey_min = np.min(FDR_hat_Storey)

    pvals_bootstrap = np.random.choice(pvals, size=(m, B))
    MSE = np.zeros((B, len_lambda), dtype=np.float32)
    for b in range(B):
        for lambda_idx, lambda_ in enumerate(lambda_arr):
            if gamma <= lambda_:
                test_stat_bootstrap = np.sum(pvals_bootstrap[:, b] > lambda_)
                pi0_hat_Storey_bootstrap = test_stat_bootstrap / (m * (1 - lambda_))
                Pr_hat_Storey_bootstrap = max(np.sum(pvals_bootstrap[:, b] <= gamma), 1) / m
                FDR_hat_Storey_bootstrap = pi0_hat_Storey_bootstrap * gamma / Pr_hat_Storey_bootstrap
            elif gamma > lambda_:
                FDR_hat_Storey_bootstrap = 1
            MSE[b, lambda_idx] = (FDR_hat_Storey_min - FDR_hat_Storey_bootstrap)**2

    opt_lambda_idx = np.argmin(np.mean(MSE, axis=0))
    # opt_lambda = lambda_arr[opt_lambda_idx]
    opt_FDR_hat_Storey = FDR_hat_Storey[opt_lambda_idx]
    return opt_FDR_hat_Storey

def compute_FDP(rejectBool_, m0):
    """
    Computing the false discovery proportion. Assuming the m0 first data
    are the true nulls.

    Inputs:
    -------
        rejectBool : ndarray, size=(m,)
            If true reject.
        m0 : integer
            The number of true nulls.

    Output:
    -------
        FDP : float
            The false discovery proportion.
    """
    R = np.sum(rejectBool_)
    RcapH = np.sum(rejectBool_[:m0])
    if R == 0:
        FDP = 0
    else:
        FDP = RcapH/R
    return FDP

def compute_FDR(rejectBool, m0, data_sims):
    """
    Monte Carlo estimate of the false discovery rate, as well as an array of
    false discovery proportions. Assuming the m0 first data are the true nulls.

    Inputs:
    -------
        rejectBool : ndarray, size=(m,)
            If true reject.
        m0 : integer
            The number of true nulls.
        data_sims : integer
            Number of simulations.

    Output:
    -------
        FDR : float
            The false discovery rate.
        FDP : ndarray, size=(data_sims,)
            The false discovery proportions.
    """
    FDP = np.zeros(data_sims, dtype=np.float64)
    for j in range(data_sims):
        FDP[j] = compute_FDP(rejectBool[j], m0)
    FDR = np.mean(FDP)
    return FDR, FDP

def compute_TDP(rejectBool_, m0, m):
    """
    Computing the true discovery proportion. Assuming the m0 first data
    are the true nulls.

    Inputs:
    -------
        rejectBool : ndarray, size=(m,)
            If true reject.
        m0 : integer
            The number of true nulls.
        m : integer
            The number of tests.

    Output:
    -------
        TDP : float
            The true discovery proportion.
    """
    m1P = m-m0
    RcapHc = np.sum(rejectBool_[m0:])
    if m == m0:
        TDP = 0
    else:
        TDP = RcapHc/m1P
    return TDP

def compute_TDR(rejectBool, m0, m, data_sims):
    """
    Monte Carlo estimate of the true discovery rate, as well as an array of
    true discovery proportions. Assuming the m0 first data are the true nulls.

    Inputs:
    -------
        rejectBool : ndarray, size=(m,)
            If true reject.
        m0 : integer
            The number of true nulls.
        m : integer
            The number of tests.
        data_sims : integer
            Number of simulations.

    Output:
    -------
        TDR : float
            The true discovery rate.
        TDP : ndarray, size=(data_sims,)
            The true discovery proportions.
    """
    TDP = np.zeros(data_sims)
    for j in range(data_sims):
        TDP[j] = compute_TDP(rejectBool[j], m0, m)
    TDR = np.mean(TDP)
    return TDR, TDP
