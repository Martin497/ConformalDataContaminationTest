#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 08:17:21 2025

This module contains some basic utility used in the experiment scripts:
    - BudgetAccuracy.py
    - BudgetAccuracyCV.py
    - ScoringAnalysis.py
    - ProposedAccuracy.py
    - RejectionThresholdAccuracy.py
"""

import numpy as np
from Benjamini_Hochberg import Benjamini_Hochberg_procedure, Storeys_correction


def simulate_contamination_beta(alpha_null, beta_null, alpha_alt, beta_alt, pi_boundary, K0, K):
    """
    Simulate an array of contamination factors from a mixture of Beta distributions.

    Inputs:
    ------
        alpha_null : float
            Parameter of null Beta distribution.
        beta_null : float
            Parameter of null Beta distribution.
        alpha_alt : float
            Parameter of alternative Beta distribution.
        beta_alt : float
            Parameter of alternative Beta distribution.
        pi_boundary : float in (0, 1)
            Contamination factor value splitting null and alternative.
        K0 : int
            Number of data agents simulated under the null.
        K : int
            Number of data agets.

    Outputs:
    --------
        pi_null_k : ndarray, size=(K0,)
            The contamination factors for the data agents under the null.
        pi_alt_k : ndarray, size=(K,)
            The contamination factors for the data agents under the alternative.
    """
    pi_null_k = np.random.beta(alpha_null, beta_null, size=K0) * pi_boundary
    pi_alt_k = pi_boundary + np.random.beta(alpha_alt, beta_alt, size=K-1-K0) * (1-pi_boundary)
    return pi_null_k, pi_alt_k

def import_data_time1(data_dict, d, K):
    """
    Import the data in round 1.

    Inputs:
    -------
        data_dict : dict
            Data dictionary as output from data_splitting method in DataHandlerModule.
        d : int
            The dimension of the features.
        K : int
            The number of data agents.

    Outputs:
    --------
        potential_train_data : ndarray, size=(mK, d)
            The features of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_labels : ndarray, size=(mK,)
            The labels of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_null_indicator : ndarray, size=(mK,)
            Indicator for which data points are contaminated among the mK data
            points acquired from the other agents in round 1 of data sharing.
            False indicates that the data point is an inlier while True indicates
            an outlier, i.e., a contaminated data point. This is used for the
            Oracle 1 baseline.
    """
    potential_train_data = np.zeros((0, d), dtype=np.float32)
    potential_train_labels = np.zeros(0, dtype=np.float32)
    potential_train_null_indicator = np.zeros(0, dtype=bool)
    for k in range(1, K):
        potential_train_data = np.concatenate((potential_train_data, data_dict[f"Agent{k}_Time{0}_test"][0]), axis=0)
        potential_train_labels = np.hstack((potential_train_labels, data_dict[f"Agent{k}_Time{0}_test"][1]))
        potential_train_null_indicator = np.hstack((potential_train_null_indicator, data_dict[f"Agent{k}_Time{0}_test"][2]))
    return potential_train_data, potential_train_labels, potential_train_null_indicator

def import_data_time2_oracle(data_dict, rho, K, potential_train_data, potential_train_labels, potential_train_null_indicator, idx_sort):
    """
    With the Oracle import the data in round 2 from the data agents with the
    smallest contamination factors.

    Inputs:
    -------
        data_dict : dict
            Data dictionary as output from data_splitting method in DataHandlerModule.
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        potential_train_data : ndarray, size=(mK, d)
            The features of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_labels : ndarray, size=(mK,)
            The labels of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_null_indicator : ndarray, size=(mK,)
            Indicator for which data points are contaminated among the mK data
            points acquired from the other agents in round 1 of data sharing.
            False indicates that the data point is an inlier while True indicates
            an outlier, i.e., a contaminated data point. This is used for the
            Oracle 1 baseline.
        idx_sort : ndarray, size=(K,)
            The index array sorting the data agents according to contamination factors.

    Outputs:
    --------
        additional_oracle_train_data : ndarray, size=(?+rho*K, d)
            The features of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing with Oracle 1.
        additional_oracle_train_labels : ndarray, size=(?+rho*K,)
            The labels of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing with Oracle 1.
        additional_oracle_train_null_indicator : ndarray, size=(?+rho*K,)
            Indicator for which data points are contaminated among the data
            points acquired from the other agents in rounds 1 and 2 of data
            sharing with Oracle 1.
    """
    additional_oracle_train_data = potential_train_data[np.invert(potential_train_null_indicator)]
    additional_oracle_train_labels = potential_train_labels[np.invert(potential_train_null_indicator)]
    additional_oracle_train_null_indicator = potential_train_null_indicator[np.invert(potential_train_null_indicator)]
    for k in range(1, K):
        if (k-1) in idx_sort[:rho]:
            additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{1}_test"][0]), axis=0)
            additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{1}_test"][1]))
            additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{1}_test"][2]))
    return additional_oracle_train_data, additional_oracle_train_labels, additional_oracle_train_null_indicator

def import_data_time2_oracleth(data_dict, pi_th, pi_k):
    """
    See import_data_time2_oracle().
    """
    K = len(pi_k)+1
    additional_oracle_train_data = np.zeros((0, data_dict[f"Agent{1}_Time{0}_test"][0].shape[1]), dtype=np.float32)
    additional_oracle_train_labels = np.zeros(0, dtype=np.float32)
    additional_oracle_train_null_indicator = np.zeros(0, dtype=bool)
    for k in range(1, K):
        additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{0}_test"][0]), axis=0)
        additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{0}_test"][1]))
        additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{0}_test"][2]))
    for k in range(1, K):
        if pi_k[k-1] <= pi_th:
            additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{1}_test"][0]), axis=0)
            additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{1}_test"][1]))
            additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{1}_test"][2]))
    return additional_oracle_train_data, additional_oracle_train_labels, additional_oracle_train_null_indicator


def import_data_time2_oracleth_(data_dict, pi_th, pi_k):
    """
    See import_data_time2_oracle().
    """
    K = len(pi_k)+1
    additional_oracle_train_data = np.zeros((0, data_dict[f"Agent{1}_Time{0}_test"][0].shape[1]), dtype=np.float32)
    additional_oracle_train_labels = np.zeros(0, dtype=np.float32)
    additional_oracle_train_null_indicator = np.zeros(0, dtype=bool)
    for k in range(1, K):
        if pi_k[k-1] <= pi_th:
            additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{0}_test"][0]), axis=0)
            additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{0}_test"][1]))
            additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{0}_test"][2]))
    for k in range(1, K):
        if pi_k[k-1] <= pi_th:
            additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{1}_test"][0]), axis=0)
            additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{1}_test"][1]))
            additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{1}_test"][2]))
    return additional_oracle_train_data, additional_oracle_train_labels, additional_oracle_train_null_indicator

def import_data_time2_all(data_dict, K, potential_train_data, potential_train_labels, potential_train_null_indicator):
    """
    Import the data in round 2 and combine with data from round 1.

    Inputs:
    -------
        data_dict : dict
            Data dictionary as output from data_splitting method in DataHandlerModule.
        K : int
            The number of data agents.
        potential_train_data : ndarray, size=(mK, d)
            The features of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_labels : ndarray, size=(mK,)
            The labels of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_null_indicator : ndarray, size=(mK,)
            Indicator for which data points are contaminated among the mK data
            points acquired from the other agents in round 1 of data sharing.
            False indicates that the data point is an inlier while True indicates
            an outlier, i.e., a contaminated data point. This is used for the
            Oracle 1 baseline.

    Outputs:
    --------
        additional_oracle_train_data : ndarray, size=(2mK, d)
            The features of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing.
        additional_oracle_train_labels : ndarray, size=(2mK,)
            The labels of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing.
        additional_oracle_train_null_indicator : ndarray, size=(2mK,)
            Indicator for which data points are contaminated among the data
            points acquired from the other agents in rounds 1 and 2 of data
            sharing.
    """
    additional_oracle_train_data = potential_train_data
    additional_oracle_train_labels = potential_train_labels
    additional_oracle_train_null_indicator = potential_train_null_indicator
    for k in range(1, K):
        additional_oracle_train_data = np.concatenate((additional_oracle_train_data, data_dict[f"Agent{k}_Time{1}_test"][0]), axis=0)
        additional_oracle_train_labels = np.hstack((additional_oracle_train_labels, data_dict[f"Agent{k}_Time{1}_test"][1]))
        additional_oracle_train_null_indicator = np.hstack((additional_oracle_train_null_indicator, data_dict[f"Agent{k}_Time{1}_test"][2]))
    return additional_oracle_train_data, additional_oracle_train_labels, additional_oracle_train_null_indicator

def import_data_time2_fixed(data_dict, rho, K, potential_train_data, potential_train_labels):
    """
    With the Random baseline import the data in round 2 from the data agents.

    Inputs:
    -------
        data_dict : dict
            Data dictionary as output from data_splitting method in DataHandlerModule.
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        potential_train_data : ndarray, size=(mK, d)
            The features of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_labels : ndarray, size=(mK,)
            The labels of the mK data points acquired from the other agents
            in round 1 of data sharing.
        potential_train_null_indicator : ndarray, size=(mK,)
            Indicator for which data points are contaminated among the mK data
            points acquired from the other agents in round 1 of data sharing.
            False indicates that the data point is an inlier while True indicates
            an outlier, i.e., a contaminated data point. This is used for the
            Oracle 1 baseline.

    Outputs:
    --------
        additional_oracle_train_data : ndarray, size=(mk+rho*K, d)
            The features of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing with Oracle 1.
        additional_oracle_train_labels : ndarray, size=(mk+rho*K,)
            The labels of the data points acquired from the other agents
            in rounds 1 and 2 of data sharing.
    """
    additional_fixed_train_data = np.copy(potential_train_data)
    additional_fixed_train_labels = np.copy(potential_train_labels)
    for k in range(1, K):
        if k <= rho:
            additional_fixed_train_data = np.concatenate((additional_fixed_train_data, data_dict[f"Agent{k}_Time{1}_test"][0]), axis=0)
            additional_fixed_train_labels = np.hstack((additional_fixed_train_labels, data_dict[f"Agent{k}_Time{1}_test"][1]))
    return additional_fixed_train_data, additional_fixed_train_labels

def choose_data_subset_oracle_time1(rho, K, m, idx_sort):
    """
    With the Oracle make an indicator array of which data agents to acquire data
    from in round 1: used in the data-dependent hyperparameter selection of
    BudgetAccuracyCV. This indicator array can then be combined with the total
    data from round 1.

    Inputs:
    -------
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.
        idx_sort : ndarray, size=(K,)
            The index array sorting the data agents according to contamination factors.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(1, K):
        if (k-1) in idx_sort[:rho]:
            indicator_arr = np.hstack((indicator_arr, np.ones(m, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(m, dtype=bool)))
    indicator_arr = np.hstack((indicator_arr, np.zeros(m*(K-1), dtype=bool)))
    return indicator_arr

def choose_data_subset_oracle(rho, K, mr0, mr1, idx_sort):
    """
    With the Oracle make an indicator array of which data agents to acquire data
    from. This indicator array can then be combined with the total data from
    rounds 1 and 2 coming from import_data_time2_all(...).

    Inputs:
    -------
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.
        idx_sort : ndarray, size=(K,)
            The index array sorting the data agents according to contamination factors.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(mr0*(K-1), dtype=bool)
    for k in range(1, K):
        if (k-1) in idx_sort[:rho]:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def choose_data_subset_oracle_th(pi_th, mr0, mr1, pi_k):
    """
    With the pi threshold Oracle make an indicator array of which data agents
    to acquire data from in round 1: used in the data-dependent hyperparameter
    selection of ProposedAccuracyCV. This indicator array can then be
    combined with the total data from round 1.

    Inputs:
    -------
        pi_th : float
            The contamination factor threshold.
        m : int
            The number of data samples per data agent per round.
        pi_k : ndarray, size=(K,)
            The array of contamination factors.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    Kminus1 = len(pi_k)
    indicator_arr = np.ones(mr0*Kminus1, dtype=bool)
    for k in range(Kminus1):
        if pi_k[k] <= pi_th:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def choose_data_subset_oracle_th_time1(pi_th, m, pi_k):
    """
    With the Oracle make an indicator array of which data agents to acquire data
    from in round 1: used in the data-dependent hyperparameter selection of
    BudgetAccuracyCV. This indicator array can then be combined with the total
    data from round 1.

    Inputs:
    -------
        pi_th : float
            The contamination factor threshold.
        m : int
            The number of data samples per data agent per round.
        pi_k : ndarray, size=(K,)
            The array of contamination factors.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    Kminus1 = len(pi_k)
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(Kminus1):
        if pi_k[k] <= pi_th:
            indicator_arr = np.hstack((indicator_arr, np.ones(m, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(m, dtype=bool)))
    indicator_arr = np.hstack((indicator_arr, np.zeros(m*Kminus1, dtype=bool)))
    return indicator_arr


def choose_data_subset_oracle_th_picky(pi_th, mr0, mr1, pi_k):
    """
    With the Oracle make an indicator array of which data agents to acquire data
    from. This indicator array can then be combined with the total data from
    rounds 1 and 2 coming from import_data_time2_all(...).

    Inputs:
    -------
        pi_th : float
            The contamination factor threshold.
        m : int
            The number of data samples per data agent per round.
        pi_sorted : ndarray, size=(K,)
            The array of contamination factors.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    Kminus1 = len(pi_k)
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(Kminus1):
        if pi_k[k] <= pi_th:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr0, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr0, dtype=bool)))
    for k in range(Kminus1):
        if pi_k[k] <= pi_th:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def choose_data_subset_fixed_time1(rho, K, m):
    """
    With the Random baseline make an indicator array of which data agents
    to acquire data from in round 1: used in the data-dependent hyperparameter
    selection of BudgetAccuracyCV. This indicator array can then be combined
    with the total data from round 1.

    Inputs:
    -------
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(1, K):
        if k <= rho:
            indicator_arr = np.hstack((indicator_arr, np.ones(m, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(m, dtype=bool)))
    indicator_arr = np.hstack((indicator_arr, np.zeros(m*(K-1), dtype=bool)))
    return indicator_arr

def choose_data_subset_fixed(rho, K, mr0, mr1):
    """
    With the Random baseline make an indicator array of which data agents to
    acquire data from. This indicator array can then be combined with the total
    data from rounds 1 and 2 coming from import_data_time2_all(...).

    Inputs:
    -------
        rho : int
            The data sharing budget.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(mr0*(K-1), dtype=bool)
    for k in range(1, K):
        if k <= rho:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def choose_data_subset_time1(rejectBool_con, K, m):
    """
    With the proposed method make an indicator array of which data agents
    to acquire data from in round 1: used in the data-dependent hyperparameter
    selection of BudgetAccuracyCV. This indicator array can then be combined
    with the total data from round 1.

    Inputs:
    -------
        rejectBool_con : ndarray, size=(K,)
            The decision of which data agents to collaborate with. True means
            rejection/not buying, while False means to collaborate.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(1, K):
        if rejectBool_con[k-1] == False:
            indicator_arr = np.hstack((indicator_arr, np.ones(m, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(m, dtype=bool)))
    indicator_arr = np.hstack((indicator_arr, np.zeros(m*(K-1), dtype=bool)))
    return indicator_arr

def choose_data_subset(rejectBool_con, K, mr0, mr1):
    """
    With the proposed method make an indicator array of which data agents to
    acquire data from. This indicator array can then be combined with the total
    data from rounds 1 and 2 coming from import_data_time2_all(...).

    Inputs:
    -------
        rejectBool_con : ndarray, size=(K,)
            The decision of which data agents to collaborate with. True means
            rejection/not buying, while False means to collaborate.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(mr0*(K-1), dtype=bool)
    for k in range(1, K):
        if rejectBool_con[k-1] == False:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def choose_data_subset_picky(rejectBool_con, K, mr0, mr1):
    """
    With the proposed method make an indicator array of which data agents to
    acquire data from. This indicator array can then be combined with the total
    data from rounds 1 and 2 coming from import_data_time2_all(...).

    Inputs:
    -------
        rejectBool_con : ndarray, size=(K,)
            The decision of which data agents to collaborate with. True means
            rejection/not buying, while False means to collaborate.
        K : int
            The number of data agents.
        m : int
            The number of data samples per data agent per round.

    Output:
    -------
        indicator_arr : ndarray, size=(2mK,)
            Indicating which data points to acquire, with True indicating
            buying and False indicating not buying.
    """
    indicator_arr = np.ones(0, dtype=bool)
    for k in range(1, K):
        if rejectBool_con[k-1] == False:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr0, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr0, dtype=bool)))
    for k in range(1, K):
        if rejectBool_con[k-1] == False:
            indicator_arr = np.hstack((indicator_arr, np.ones(mr1, dtype=bool)))
        else:
            indicator_arr = np.hstack((indicator_arr, np.zeros(mr1, dtype=bool)))
    return indicator_arr

def compute_FDP_TDP(rejectBool_, null_indicator):
    """
    Compute false discovery proportion (FDP) and true discovery proportion (TDP).

    Inputs:
    -------
        rejectBool : ndarray, size=(m,)
            True means to reject.
        null_indicator : ndarray, size=(m,)
            True indicates and outlier, and False an inlier.

    Outputs:
    --------
        FDP : float
            False discovery proportion.
        TDP : float
            True discovery proportion.
    """
    R = np.sum(rejectBool_)
    RcapH = np.sum(rejectBool_[np.invert(null_indicator)])
    if R == 0:
        FDP = 0
    else:
        FDP = RcapH/R

    K1P = np.sum(null_indicator)
    RcapHc = np.sum(rejectBool_[null_indicator])
    if K1P == 0:
        TDP = 0
    else:
        TDP = RcapHc/K1P
    return FDP, TDP

def COD_fit_score(potential_train_data, potential_train_labels, train_data, train_labels,
                  in_conformal_pvalues, zeta, beta, model_handler, null_indicator, test_data, test_labels, feature_dim):
    """
    Run conformal outlier detection, fit the model, and compute test score.
    """
    M = len(in_conformal_pvalues)
    rejectBoolCOD = np.zeros(M, dtype=bool)
    if beta > 0:
        classes = np.unique(potential_train_labels)
        for class_idx, class_ in enumerate(classes):
            indicator_arr = potential_train_labels == class_
            class_conformal_pvalues = in_conformal_pvalues[indicator_arr]
            class_M = len(class_conformal_pvalues)
            _, class_M0_hat = Storeys_correction(zeta, class_conformal_pvalues, class_M)
            rejectBoolCOD[indicator_arr] \
                = Benjamini_Hochberg_procedure(class_conformal_pvalues, beta, class_M0_hat, class_M)

    FDP, TDP = compute_FDP_TDP(rejectBoolCOD, null_indicator)

    additional_train_data = potential_train_data[np.invert(rejectBoolCOD)]
    additional_train_labels = potential_train_labels[np.invert(rejectBoolCOD)]
    train_data = np.concatenate((train_data, additional_train_data), axis=0)
    train_labels = np.hstack((train_labels, additional_train_labels))
    model_handler.fit(train_data, train_labels) # SLOW
    model_scores = model_handler.score(test_data, test_labels, feature_dim)
    return model_scores, train_data.shape[0], FDP, TDP

def proposed(rejectBool_, K, mr0, mr1, allin_conformal_pvalues, allin_train_data, allin_train_labels, train_data, train_labels,
             zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels, feature_dim):
    """
    The proposed method. Chooses the relevant data, runs conformal outlier detection,
    fits the model, and computes the test score.
    """
    indicator_arr = choose_data_subset(rejectBool_, K, mr0, mr1)
    conformal_pvalues = allin_conformal_pvalues[indicator_arr]
    potential_train_data, potential_train_labels \
        = allin_train_data[indicator_arr], allin_train_labels[indicator_arr]
    null_indicator = allin_train_null_indicator[indicator_arr]
    return COD_fit_score(potential_train_data, potential_train_labels, train_data, train_labels,
                         conformal_pvalues, zeta, beta, model_handler, null_indicator, test_data, test_labels, feature_dim)

def proposed_picky(rejectBool_, K, mr0, mr1, allin_conformal_pvalues, allin_train_data, allin_train_labels, train_data, train_labels,
             zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels, feature_dim):
    """
    The proposed method. Chooses the relevant data, runs conformal outlier detection,
    fits the model, and computes the test score.
    """
    indicator_arr = choose_data_subset_picky(rejectBool_, K, mr0, mr1)
    conformal_pvalues = allin_conformal_pvalues[indicator_arr]
    potential_train_data, potential_train_labels \
        = allin_train_data[indicator_arr], allin_train_labels[indicator_arr]
    null_indicator = allin_train_null_indicator[indicator_arr]
    return COD_fit_score(potential_train_data, potential_train_labels, train_data, train_labels,
                         conformal_pvalues, zeta, beta, model_handler, null_indicator, test_data, test_labels, feature_dim)

def compute_conformal_scores(score_name, score_handler, data_dict, allin_train_data, allin_train_labels,
                             AdaDetect_individual=False, mr0=None, mr1=None, K=None):
    """
    """
    n = len(data_dict["Agent0_calibration"][0])
    if (score_name == "OCSVM") or (score_name == "labelOCSVM") or (score_name == "IsolationForest") or (score_name == "LocalOutlierFactor") \
    or (score_name == "labelIsolationForest") or (score_name == "labelLocalOutlierFactor") or (score_name == "OCSVMXY") \
    or (score_name == "IsolationForestXY") or (score_name == "LocalOutlierFactorXY") \
    or (score_name == "AutoencoderXY") or (score_name == "Autoencoder") or (score_name == "labelAutoencoder"):
        score_handler.fit(data_dict["Agent0_train"][0], data_dict["Agent0_train"][1])
        calibration_scores = score_handler.score(data_dict["Agent0_calibration"][0], data_dict["Agent0_calibration"][1])
        allin_data_scores = score_handler.score(allin_train_data, allin_train_labels)
    elif (score_name == "AdaDetect"):
        if AdaDetect_individual is False:
            scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data)
            calibration_scores, allin_data_scores = scores[:n], scores[n:]
        elif AdaDetect_individual is True:
            calibration_scores_temp = np.zeros(2*K*n, dtype=np.float32)
            allin_data_scores_temp = np.hstack((np.zeros(K*mr0, dtype=np.float32), np.zeros(K*mr1, dtype=np.float32)))
            for k in range(K):
                scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data[k*mr0:(k+1)*mr0])
                calibration_scores_temp[k*n:(k+1)*n], allin_data_scores_temp[k*mr0:(k+1)*mr0] = scores[:n], scores[n:]
            for k in range(K, 2*K):
                scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data[k*mr1:(k+1)*mr1])
                calibration_scores_temp[k*n:(k+1)*n], allin_data_scores_temp[k*mr1:(k+1)*mr1] = scores[:n], scores[n:]
    elif (score_name == "labelAdaDetect") or ((score_name == "AdaDetectXY")):
        if AdaDetect_individual is False:
            scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data,
                                                  data_dict["Agent0_train"][1], data_dict["Agent0_calibration"][1], allin_train_labels)
            calibration_scores, allin_data_scores = scores[:n], scores[n:]
        elif AdaDetect_individual is True:
            calibration_scores_temp = np.zeros(2*K*n, dtype=np.float32)
            allin_data_scores_temp = np.hstack((np.zeros(K*mr0, dtype=np.float32), np.zeros(K*mr1, dtype=np.float32)))
            for k in range(K):
                scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data[k*mr0:(k+1)*mr0],
                                                      data_dict["Agent0_train"][1], data_dict["Agent0_calibration"][1], allin_train_labels[k*mr0:(k+1)*mr0])
                calibration_scores_temp[k*n:(k+1)*n], allin_data_scores_temp[k*mr0:(k+1)*mr0] = scores[:n], scores[n:]
            for k in range(K, 2*K):
                scores = score_handler.adaptive_score(data_dict["Agent0_train"][0], data_dict["Agent0_calibration"][0], allin_train_data[k*mr1:(k+1)*mr1],
                                                      data_dict["Agent0_train"][1], data_dict["Agent0_calibration"][1], allin_train_labels[k*mr1:(k+1)*mr1])
                calibration_scores_temp[k*n:(k+1)*n], allin_data_scores_temp[k*mr1:(k+1)*mr1] = scores[:n], scores[n:]
    return calibration_scores, allin_data_scores
