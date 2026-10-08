#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:22:05 2025

In this script the conformal data contamination tests are used for a collaborative
data sharing system. Relevant hyperparameters of each method is found by
cross-validation using data from the first round.

Abbreviations:
    - Conformal outlier detection (COD)
    - Data subset selection (DSS)
    - Benjamini-Hochberg (BH)

Baseline methods:
    - baseline [random selection under a fixed budget]
    - baseline_COD [random selection under a fixed budget using COD for DSS]
    - oracle [choose collaborators with smallest pi_k under a fixed budget]
    - oracleth [choose collaborators as those with pi_k <= pi_th in second round]
    - oracle_COD [choose collaborators with smallest pi_k under a fixed budget using COD for DSS]
    - oracle_CODth [choose collaborators as those with pi_k <= pi_th using COD for DSS]
    - oracle_CODth_picky [choose collaborators as those with pi_k <= pi_th using COD for DSS;
                          do not use any data from data agents with pi_k > pi_th]

Proposed methods:
    - proposed [choose collaborators as those which are likely pi_k <= pi_th, decisions made
                by the BH procedure at level alpha using also COD for DSS]
    - proposed_picky [choose collaborators as those which are likely pi_k <= pi_th with rejection
                      decisions made by the BH procedure at level alpha using also COD for DSS;
                      do not use any data from data agents with pi_k > pi_th]
 """


import numpy as np

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import utilities as ut

from ConformalScoreModule import ConformalScore
from DataHandlerModule import DataHandler
from ConformalContaminationTestModule import ConformalContaminationTest
from SupervisedMachineLearningModule import SupervisedMachineLearning
from Benjamini_Hochberg import Benjamini_Hochberg_procedure, Storeys_correction


def main():
    import warnings 
    warnings.filterwarnings("ignore")

    seed = 10
    np.random.seed(seed)
    sims = 50
    randomize = True
    setupname = "retina_AdaDetect_SVC_SVC"
    setupname_list = setupname.split("_")
    savename = "temp"

    ### Scenario settings ###
    if setupname_list[0] == "retinalOCT":
        data_name = f"{setupname_list[0]}_{setupname_list[1]}"
        agent_splitting_type = f"{setupname_list[0]}"
    elif setupname_list[0] == "retina":
        data_name = f"{setupname_list[0]}"
        agent_splitting_type = f"{setupname_list[0]}"
    elif setupname_list[0] == "MNIST":
        data_name = f"{setupname_list[0]}_{setupname_list[1]}"
        if setupname_list[2] == "ln":
            agent_splitting_type = "label_noise"
        elif setupname_list[2] == "fn":
            agent_splitting_type = "feature_noise"
    elif setupname_list[0] == "FEMNIST":
        data_name = f"{setupname_list[0]}_{setupname_list[1]}"
        agent_splitting_type = "femnist"
    elif setupname_list[0] == "WBC":
        data_name = f"{setupname_list[0]}_{setupname_list[1]}"
        agent_splitting_type = f"{setupname_list[0]}"

    if data_name == "MNIST_partial":
        classes = [1, 4, 7]
        alt_classes = None
        feature_dim = (28, 28, 1)
    elif data_name == "FEMNIST_partial":
        classes = [10, 11, 12] # ABC
        alt_classes = [36, 37, 38] # abc
        feature_dim = (28, 28, 1)
    elif (data_name == "retina") or (data_name == "EyePACS") or (data_name == "EyePACS_wtest"):
        classes = [0, 1, 2, 3, 4]
        alt_classes = None
        feature_dim = (28, 28, 3)
    elif data_name == "retinalOCT":
        classes = [0, 1, 2, 3]
        alt_classes = None
        feature_dim = (28, 28, 1)
    elif data_name == "retinalOCT_partial":
        classes = [0, 3]
        alt_classes = None
        feature_dim = (28, 28, 1)
    elif data_name == "WBC_partial":
        classes = [0, 1, 4, 5]
        alt_classes = None
        feature_dim = (28, 28, 3)
    shape = np.prod(feature_dim)

    include = ["baseline_COD", "oracle_CODth", "proposed"]

    K = 10+1 # K - 1 other agents
    T = 2
    ell0 = 60
    n = 40
    m = 40

    ### Feature noise ###
    bin_p = None
    bin_p_alt = None
    repeats = None
    block_size = None
    std = 0
    distance_cov = 0
    std_alt = 0.3
    distance_cov_alt = 0.1
    if agent_splitting_type == "feature_noise":
        feature_noise_type = "GRF_norm"
        feature_noise = "test"
        alt_noise_type = "GRF_norm"
    else:
        feature_noise_type = "GRF_norm"
        feature_noise = "train_test"
        alt_noise_type = None

    ### Contamination ###
    pi_choice_type = "uniform"
    pi_param = -1
    K0 = -1

    ### Hyperparameters ###
    score_name = setupname_list[-3]
    if (score_name == "AdaDetect") or (score_name == "labelAdaDetect"):
        if setupname_list[-2] == "LR":
            AdaDetect_classifier = "LogisticRegression"
        elif setupname_list[-2] == "SVC":
            AdaDetect_classifier = "SVC"
        AdaDetect_individual = False
    elif score_name == "OCSVM":
        assert setupname_list[-2] == "", ""

    if setupname_list[-1] == "SVC":
        model_name = "SVC"
        model_kwargs = {"kernel": "rbf", "gamma": "auto"}
    elif setupname_list[-1] == "LR":
        model_name = "LogisticRegression"
        model_kwargs = dict()
    elif setupname_list[-1] == "MLP":
        model_name = "MLPClassifier"
        model_kwargs = dict()
    elif setupname_list[-1] == "GBC":
        model_name = "GradientBoostingClassifier"
        model_kwargs = dict()

    significance_levels = [0.05, 0.2, 0.5, 0.7]
    num_rejection_thresholds = len(significance_levels)

    use_Storey = True
    gamma = 0.5 # Storey's hyperparameter conformal contamination test

    pi_th_arr = np.linspace(0, 0.7, num=8) # Threshold conformal contamination test
    len_pi_th_arr = len(pi_th_arr)

    pi_th_arr_oracle = np.linspace(0, 1, num=21) # Threshold conformal contamination test
    len_pi_th_arr_oracle = len(pi_th_arr_oracle)

    rho_arr = np.arange(K)
    C1, C2 = np.meshgrid(pi_th_arr, significance_levels)
    hyper_grid_proposed = np.stack((C1.flatten(), C2.flatten()), axis=1)

    beta = -1 # Significance level conformal outlier detection
    zeta = -1 # Storey's hyperparameter conformal outlier detection

    lambda_ = n//8 / (n+1)
    i0_param = 3
    i0_arr = [m//i0_param for pi_th in pi_th_arr]

    with open(f"ProposedAccuracyCV/{savename}.txt", "w") as file:
        file.write(f"sims: {sims:d}\n")
        file.write(f"seed: {seed:d}\n")
        file.write(f"K: {K-1:d}\n")
        file.write(f"ell: {ell0:d}\n")
        file.write(f"n: {n:d}\n")
        file.write(f"m: {m:d}\n")
        file.write(f"agent splitting type: {agent_splitting_type}\n")
        file.write(f"feature noise type: {feature_noise_type}\n")
        file.write(f"std: {std:.2f}\n")
        file.write(f"distance cov: {distance_cov:.2f}\n")
        if agent_splitting_type == "feature_noise":
            if alt_noise_type == "GRF" or alt_noise_type == "GRF_norm":
                file.write(f"std alt: {std_alt:.2f}\n")
                file.write(f"distance cov alt: {distance_cov_alt:.2f}\n")
            elif alt_noise_type == "bin_p":
                file.write(f"bin p alt: {bin_p_alt:.2f}\n")
        file.write(f"pi model: {pi_choice_type}\n")
        if pi_choice_type == "fixed" or pi_choice_type == "beta":
            file.write(f"pi parameter: {pi_param:.2f}\n")
            file.write(f"K0: {K0:d}\n")
        file.write(f"gamma: {gamma:.2f}\n")
        file.write(f"beta: {beta:.2f}\n")
        file.write(f"zeta: {zeta:.2f}\n")
        file.write(f"lambda: {lambda_:.2f}\n")
        file.write(f"i0 parameter: {i0_param:.2f}\n")
        file.write(f"Model type: {model_name}\n")
        file.write(f"Score type: {score_name}\n")
        if score_name == "AdaDetect":
            file.write(f"AdaDetect classifier: {AdaDetect_classifier}\n")
            file.write(f"AdaDetect individual: {AdaDetect_individual}\n")
        if data_name == "MNIST_partial":
            file.write("Classes: ")
            for class_ in classes:
                file.write(f"{class_:d}, ")
            file.write("\n")
        if data_name == "FEMNIST_partial":
            file.write("Null classes: ")
            for class_ in classes:
                file.write(f"{class_:d}, ")
            file.write("\n")
            file.write("Alt classes: ")
            for class_ in alt_classes:
                file.write(f"{class_:d}, ")
            file.write("\n")
        file.write(f"Use Storey: {use_Storey}\n")
        for inc in include:
            file.write(f"{inc}, ")
        file.write("\n")


    ### Class preparation ###
    data_handler = DataHandler(data_name, feature_noise=feature_noise, feature_noise_type=feature_noise_type,
                               classes=classes, alt_classes=alt_classes, bin_p=bin_p, repeats=repeats, block_size=block_size,
                               std=std, distance_cov=distance_cov)
    score_handler = ConformalScore(score_name, labels=classes, classifier=AdaDetect_classifier,
                                   shape=shape, use_PCA=False, n_components=None,
                                   feature_dim = feature_dim)
    test_handler = ConformalContaminationTest()
    model_handler = SupervisedMachineLearning(model_name, **model_kwargs)

    ### Organize test data ###
    test_data, test_labels = data_handler.return_test_data()

    ### Loop begin ###
    pi_true = np.zeros((sims, K-1))
    pcon_total = np.zeros((sims, len(pi_th_arr), K-1, 4), dtype=np.float32)
    CCTest_data_proposed = np.zeros((sims, 4, 2), dtype=np.float32)

    if "baseline_COD" in include:
        model_scores_baselines_COD = np.zeros(sims, dtype=np.float32)
        bought_data_baselines_COD = np.zeros(sims, dtype=np.float32)
        used_data_baselines_COD = np.zeros(sims, dtype=np.float32)
        CODTest_data_baselines_COD = np.zeros((sims, 2), dtype=np.float32)

    if "oracle_COD" in include:
        model_scores_partoracles = np.zeros(sims, dtype=np.float32)
        bought_data_partoracles = np.zeros(sims, dtype=np.float32)
        used_data_partoracles = np.zeros(sims, dtype=np.float32)
        CODTest_data_partoracles = np.zeros((sims, 2), dtype=np.float32)

    if "oracle_CODth" in include:
        model_scores_partoracles_th = np.zeros(sims, dtype=np.float32)
        bought_data_partoracles_th = np.zeros(sims, dtype=np.float32)
        used_data_partoracles_th = np.zeros(sims, dtype=np.float32)
        CODTest_data_partoracles_th = np.zeros((sims, 2), dtype=np.float32)

    if "proposed" in include:
        model_scores_proposed = np.zeros((sims, 4), dtype=np.float32)
        bought_data_proposed = np.zeros((sims, 4), dtype=np.float32)
        used_data_proposed = np.zeros((sims, 4), dtype=np.float32)
        CODTest_data_proposed = np.zeros((sims, 4, 2), dtype=np.float32)

    for sim in range(sims):
        print("Simulation:", sim+1, "/", sims)

        if pi_choice_type == "fixed":
            pi_null = pi_param * np.ones(K0)
            pi_alt = 1-pi_param * np.ones(K-1-K0)
            randomize_array = np.random.permutation(K-1)
            pi_k = np.hstack((pi_null, pi_alt))[randomize_array]
        elif pi_choice_type == "beta":
            pi_boundary = pi_param
            pi_null, pi_alt = ut.simulate_contamination_beta(1, 3, 5, 1, pi_boundary, K0, K)
            randomize_array = np.random.permutation(K-1)
            pi_k = np.hstack((pi_null, pi_alt))[randomize_array]
        elif pi_choice_type == "uniform":
            pi_k = np.random.uniform(low=0, high=1, size=(K-1))
        elif pi_choice_type == "discrete_uniform":
            pi_k = np.random.randint(low=0, high=11, size=(K-1))/10
        pi_true[sim, :] = pi_k
        print(pi_k)

        data_dict = data_handler.data_splitting(K=K, K0=None, agent_splitting_type=agent_splitting_type, ell0=ell0, n0=n,
                                                pi_k=pi_k, randomize=randomize, mr0=m, mr1=m, T=T, bin_p=bin_p, repeats=repeats, block_size=block_size,
                                                std=std, distance_cov=distance_cov, std_alt=std_alt, distance_cov_alt=distance_cov_alt,
                                                alt_noise_type=alt_noise_type, bin_p_alt=bin_p_alt)

        ### Organize data (part 1) ###
        train_data = data_dict["Agent0_train"][0]
        train_data = np.concatenate((train_data, data_dict["Agent0_calibration"][0]), axis=0)
        train_labels = data_dict["Agent0_train"][1]
        train_labels = np.hstack((train_labels, data_dict["Agent0_calibration"][1]))

        potential_train_data, potential_train_labels, potential_train_null_indicator = ut.import_data_time1(data_dict, data_handler.d, K)

        ### All the data ###
        allin_train_data, allin_train_labels, allin_train_null_indicator \
            = ut.import_data_time2_all(data_dict, K, potential_train_data, potential_train_labels, potential_train_null_indicator)

        ### Conformal scores and p-values ###
        calibration_scores, allin_data_scores = ut.compute_conformal_scores(score_name, score_handler, data_dict,
                                                                            allin_train_data, allin_train_labels,
                                                                            AdaDetect_individual, m)
        allin_conformal_pvalues = test_handler.compute_conformal_pvalues(calibration_scores, allin_data_scores)

        ### Baseline: no data sharing ###
        subtrain_data = data_dict["Agent0_train"][0]
        subtrain_labels = data_dict["Agent0_train"][1]
        validation_data = data_dict["Agent0_calibration"][0]
        validation_labels = data_dict["Agent0_calibration"][1]
        model_handler.fit(subtrain_data, subtrain_labels)
        noshare_validation_score = model_handler.score(validation_data, validation_labels)


        ### Baseline : rho-fixed data sharing with conformal outlier detection ###
        if "baseline_COD" in include:
            baseline_validation_scores = np.zeros(K, dtype=np.float32)
            for rho in range(K):
                print("\t Random (COD):", rho, "/", K-1)
                if rho > 0:
                    temp_indicator_arr = ut.choose_data_subset_fixed_time1(rho, K, m)
                    temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
                    temp_potential_train_data, temp_potential_train_labels \
                        = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
                    temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]
                    M = len(temp_conformal_pvalues)
                    rejectBoolCOD = np.zeros(M, dtype=bool)
                    if beta > 0:
                        classes = np.unique(temp_potential_train_labels)
                        for class_idx, class_ in enumerate(classes):
                            indicator_arr = temp_potential_train_labels == class_
                            class_conformal_pvalues = temp_conformal_pvalues[indicator_arr]
                            class_M = len(class_conformal_pvalues)
                            _, class_M0_hat = Storeys_correction(zeta, class_conformal_pvalues, class_M)
                            rejectBoolCOD[indicator_arr] \
                                = Benjamini_Hochberg_procedure(class_conformal_pvalues, beta, class_M0_hat, class_M)
                    additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                    additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                    temp_train_data = np.concatenate((subtrain_data, additional_train_data), axis=0)
                    temp_train_labels = np.hstack((subtrain_labels, additional_train_labels))
                    model_handler.fit(temp_train_data, temp_train_labels)
                    baseline_validation_scores[rho] = model_handler.score(validation_data, validation_labels)
                else:
                    baseline_validation_scores[rho] = noshare_validation_score

            rho_max = int(rho_arr[np.argmax(baseline_validation_scores)])

            temp_indicator_arr = ut.choose_data_subset_fixed(rho_max, K, m, m)
            temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
            temp_potential_train_data, temp_potential_train_labels \
                = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
            temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]

            bought_data_baselines_COD[sim] = m*(K-1) + m*rho_max
            model_scores_baselines_COD[sim], used_data_baselines_COD[sim], \
            CODTest_data_baselines_COD[sim, 0], CODTest_data_baselines_COD[sim, 1] = \
                ut.COD_fit_score(temp_potential_train_data, temp_potential_train_labels, train_data, train_labels, temp_conformal_pvalues,
                                 zeta, beta, model_handler, temp_null_indicator, test_data, test_labels, feature_dim)


        ### Baseline: oracle with conformal outlier detection ###
        if "oracle_COD" in include:
            sort_idx = np.argsort(pi_k)
            bestCOD_validation_scores = np.zeros(K, dtype=np.float32)
            for rho in range(K):
                print("\t Oracle all data:", rho, "/", K-1)
                if rho > 0:
                    temp_indicator_arr = ut.choose_data_subset_oracle_time1(rho, K, m, sort_idx)
                    temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
                    temp_potential_train_data, temp_potential_train_labels \
                        = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
                    temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]
                    M = len(temp_conformal_pvalues)
                    rejectBoolCOD = np.zeros(M, dtype=bool)
                    if beta > 0:
                        classes = np.unique(temp_potential_train_labels)
                        for class_idx, class_ in enumerate(classes):
                            indicator_arr = temp_potential_train_labels == class_
                            class_conformal_pvalues = temp_conformal_pvalues[indicator_arr]
                            class_M = len(class_conformal_pvalues)
                            _, class_M0_hat = Storeys_correction(zeta, class_conformal_pvalues, class_M)
                            rejectBoolCOD[indicator_arr] \
                                = Benjamini_Hochberg_procedure(class_conformal_pvalues, beta, class_M0_hat, class_M)
                    additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                    additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                    temp_train_data = np.concatenate((subtrain_data, additional_train_data), axis=0)
                    temp_train_labels = np.hstack((subtrain_labels, additional_train_labels))
                    model_handler.fit(temp_train_data, temp_train_labels)
                    bestCOD_validation_scores[rho] = model_handler.score(validation_data, validation_labels)
                else:
                    bestCOD_validation_scores[rho] = noshare_validation_score

            rho_max = int(rho_arr[np.argmax(bestCOD_validation_scores)])

            temp_indicator_arr = ut.choose_data_subset_oracle(rho_max, K, m, m, sort_idx)
            temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
            temp_potential_train_data, temp_potential_train_labels \
                = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
            temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]

            bought_data_partoracles[sim] = m*(K-1) + m*rho_max
            model_scores_partoracles[sim], used_data_partoracles[sim], \
            CODTest_data_partoracles[sim, 0], CODTest_data_partoracles[sim, 1] = \
                ut.COD_fit_score(temp_potential_train_data, temp_potential_train_labels, train_data, train_labels, temp_conformal_pvalues,
                                 zeta, beta, model_handler, temp_null_indicator, test_data, test_labels, feature_dim)

        ### Baseline: oracle with conformal outlier detection ###
        if "oracle_CODth" in include:
            sort_idx = np.argsort(pi_k)
            bestCODth_validation_scores = np.zeros(len(pi_th_arr_oracle), dtype=np.float32)
            for pi_th_idx, pi_th in enumerate(pi_th_arr_oracle):
                print("\t Oracle threshold all data:", pi_th_idx+1, "/", len_pi_th_arr_oracle)
                temp_indicator_arr = ut.choose_data_subset_oracle_th_time1(pi_th, m, pi_k)
                temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
                temp_potential_train_data, temp_potential_train_labels \
                    = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
                temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]
                M = len(temp_conformal_pvalues)
                rejectBoolCOD = np.zeros(M, dtype=bool)
                if beta > 0:
                    classes = np.unique(temp_potential_train_labels)
                    for class_idx, class_ in enumerate(classes):
                        indicator_arr = temp_potential_train_labels == class_
                        class_conformal_pvalues = temp_conformal_pvalues[indicator_arr]
                        class_M = len(class_conformal_pvalues)
                        _, class_M0_hat = Storeys_correction(zeta, class_conformal_pvalues, class_M)
                        rejectBoolCOD[indicator_arr] \
                            = Benjamini_Hochberg_procedure(class_conformal_pvalues, beta, class_M0_hat, class_M)
                additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                temp_train_data = np.concatenate((subtrain_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((subtrain_labels, additional_train_labels))
                model_handler.fit(temp_train_data, temp_train_labels)
                bestCODth_validation_scores[pi_th_idx] = model_handler.score(validation_data, validation_labels)

            pi_th_max_idx = np.argmax(bestCODth_validation_scores)
            pi_th_max = pi_th_arr_oracle[pi_th_max_idx]

            temp_indicator_arr = ut.choose_data_subset_oracle_th(pi_th_max, m, m, pi_k)
            temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
            temp_potential_train_data, temp_potential_train_labels \
                = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
            temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]

            bought_data_partoracles_th[sim] = m*(K-1) + m*np.sum(pi_k <= pi_th_max)
            model_scores_partoracles_th[sim], used_data_partoracles_th[sim], \
            CODTest_data_partoracles_th[sim, 0], CODTest_data_partoracles_th[sim, 1] = \
                ut.COD_fit_score(temp_potential_train_data, temp_potential_train_labels, train_data, train_labels, temp_conformal_pvalues,
                                 zeta, beta, model_handler, temp_null_indicator, test_data, test_labels, feature_dim)


        ### Run conformal contamination test ###
        test_scores = np.zeros((K-1, m), dtype=np.float64)
        for k in range(1, K):
            # test_scores[k-1] = score_handler.score(data_dict[f"Agent{k}_Time{0}_test"][0], data_dict[f"Agent{k}_Time{0}_test"][1])
            test_scores[k-1] = allin_data_scores[(k-1)*m:k*m]

        pcon = np.zeros((len(pi_th_arr), K-1, 4), dtype=np.float32)
        for pi_th_idx, pi_th in enumerate(pi_th_arr):
            for k in range(1, K):
                pcon[pi_th_idx, k-1, 0], pcon[pi_th_idx, k-1, 1], pcon[pi_th_idx, k-1, 2], pcon[pi_th_idx, k-1, 3] \
                    = test_handler.all_conformal_contamination_tests(calibration_scores, test_scores[k-1],
                      pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0_arr[pi_th_idx])
                pcon_total[sim] = pcon
                # print(pcon)

        rejectBoolCCTest = np.zeros((len_pi_th_arr, num_rejection_thresholds, 4, K-1), dtype=bool)
        for idx_alpha, alpha in enumerate(significance_levels):
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for idx in range(4):
                    pvals = pcon[pi_th_idx, :, idx]
                    if use_Storey is True:
                        _, K0_hat = Storeys_correction(gamma, pvals, K-1)
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)
                    elif use_Storey is False:
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K-1, K-1)
                    rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :] = rejectBool_

        if "proposed" in include:
            ### Hyperparameter selection ###
            Proposed_validation_scores = np.zeros((len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.float32)
            for idx_alpha, alpha in enumerate(significance_levels):
                for pi_th_idx, pi_th in enumerate(pi_th_arr):
                    print(f"\t Proposed: ({idx_alpha+1}, {pi_th_idx+1}) / ({num_rejection_thresholds}, {len_pi_th_arr})")
                    for idx2 in range(4):
                        temp_indicator_arr = ut.choose_data_subset_time1(rejectBoolCCTest[pi_th_idx, idx_alpha, idx2, :], K, m)
                        temp_conformal_pvalues = allin_conformal_pvalues[temp_indicator_arr]
                        temp_potential_train_data, temp_potential_train_labels \
                            = allin_train_data[temp_indicator_arr], allin_train_labels[temp_indicator_arr]
                        temp_null_indicator = allin_train_null_indicator[temp_indicator_arr]
                        M = len(temp_conformal_pvalues)
                        rejectBoolCOD = np.zeros(M, dtype=bool)
                        if beta > 0:
                            classes = np.unique(temp_potential_train_labels)
                            for class_idx, class_ in enumerate(classes):
                                indicator_arr = temp_potential_train_labels == class_
                                class_conformal_pvalues = temp_conformal_pvalues[indicator_arr]
                                class_M = len(class_conformal_pvalues)
                                _, class_M0_hat = Storeys_correction(zeta, class_conformal_pvalues, class_M)
                                rejectBoolCOD[indicator_arr] \
                                    = Benjamini_Hochberg_procedure(class_conformal_pvalues, beta, class_M0_hat, class_M)
                        additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                        additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                        temp_train_data = np.concatenate((subtrain_data, additional_train_data), axis=0)
                        temp_train_labels = np.hstack((subtrain_labels, additional_train_labels))
                        model_handler.fit(temp_train_data, temp_train_labels)
                        Proposed_validation_scores[pi_th_idx, idx_alpha, idx2] = model_handler.score(validation_data, validation_labels)
                    else:
                        Proposed_validation_scores[pi_th_idx, idx_alpha, idx2] = noshare_validation_score

            for idx2 in range(4):
                hyper_max = hyper_grid_proposed[np.argmax(Proposed_validation_scores[:, :, idx2])]
                pi_th_max, alpha_max = hyper_max[0], hyper_max[1]
                pi_th_max_idx = np.argwhere(pi_th_arr == pi_th_max)[0, 0]
                alpha_max_idx = np.argwhere(significance_levels == alpha_max)[0, 0]

                CCTest_data_proposed[sim, idx2, 0], CCTest_data_proposed[sim, idx2, 1] \
                    = ut.compute_FDP_TDP(rejectBoolCCTest[pi_th_max_idx, alpha_max_idx, idx2, :], pi_k>pi_th_max)

                bought_data_proposed[sim, idx2] = m*(K-1) + m*((K-1) - np.sum(rejectBoolCCTest[pi_th_max_idx, alpha_max_idx, idx, :]))
                model_scores_proposed[sim, idx2], used_data_proposed[sim, idx2], \
                CODTest_data_proposed[sim, idx2, 0], CODTest_data_proposed[sim, idx2, 1] \
                    = ut.proposed(rejectBoolCCTest[pi_th_max_idx, alpha_max_idx, idx2, :], K, m, m, allin_conformal_pvalues, allin_train_data, allin_train_labels, train_data, train_labels,
                                  zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels, feature_dim)

    kwds = {"pi_true": pi_true,
            "pcon_total": pcon_total,
            "CCTest_data_proposed": CCTest_data_proposed}

    if "baseline_COD" in include:
        kwds.update({"model_scores_baselines_COD": model_scores_baselines_COD,
                     "bought_data_baselines_COD": bought_data_baselines_COD,
                     "used_data_baselines_COD": used_data_baselines_COD,
                     "CODTest_data_baselines_COD": CODTest_data_baselines_COD})
        print("baseline_COD:", np.mean(model_scores_baselines_COD))

    if "oracle_COD" in include:
        kwds.update({"model_scores_partoracles": model_scores_partoracles,
                      "bought_data_partoracles": bought_data_partoracles,
                      "used_data_partoracles": used_data_partoracles,
                      "CODTest_data_partoracles": CODTest_data_partoracles})
        print("oracle_COD:", np.mean(model_scores_partoracles))

    if "oracle_CODth" in include:
        kwds.update({"model_scores_partoracles_th": model_scores_partoracles_th,
                     "bought_data_partoracles_th": bought_data_partoracles_th,
                     "used_data_partoracles_th": used_data_partoracles_th,
                     "CODTest_data_partoracles_th": CODTest_data_partoracles_th})
        print("oracle_CODth:", np.mean(model_scores_partoracles_th))

    if "proposed" in include:
        kwds.update({"model_scores_proposed": model_scores_proposed,
                     "bought_data_proposed": bought_data_proposed,
                     "used_data_proposed": used_data_proposed,
                     "CODTest_data_proposed": CODTest_data_proposed})
        print("proposed:", np.mean(model_scores_proposed, axis=0))

    np.savez(f"ProposedAccuracyCV/{savename}.npz", **kwds)


if __name__ == "__main__":
    main()