#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:33:28 2025

In this script the conformal data contamination tests are used for a collaborative
data sharing system. The collaborating agents are selected based on the proposed method,
given a threshold on the contamination factor and a significance level.

Abbreviations:
    - Conformal outlier detection (COD)
    - Data subset selection (DSS)
    - Benjamini-Hochberg (BH)

Baseline methods:
    - baseline [random selection under a fixed budget]
    - baseline_COD [random selection under a fixed budget using COD for DSS]
    - oracle [choose collaborators with smallest pi_k under a fixed budget]
    - oracleth [choose collaborators as those with pi_k <= pi_th]
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
import matplotlib.pyplot as plt


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
    plt.style.use("seaborn-v0_8-whitegrid")
    fsize = (9.6, 5.76)
    import warnings 
    warnings.filterwarnings("ignore")

    seed = 11
    np.random.seed(seed)
    sims = 40
    randomize = True
    savename = "cifar10_brightness_K50_AdaDetect_PCALR_LR"
    savename_list = savename.split("_")

    ### Scenario settings ###
    if savename_list[0] == "retinalOCT":
        data_name = f"{savename_list[0]}_{savename_list[1]}"
        agent_splitting_type = f"{savename_list[0]}"
    elif savename_list[0] == "retina":
        data_name = f"{savename_list[0]}"
        agent_splitting_type = f"{savename_list[0]}"
    elif savename_list[0] == "MNIST":
        data_name = f"{savename_list[0]}_{savename_list[1]}"
        if savename_list[2] == "ln":
            agent_splitting_type = "label_noise"
        elif savename_list[2] == "fn":
            agent_splitting_type = "feature_noise"
    elif savename_list[0] == "FEMNIST":
        data_name = f"{savename_list[0]}_{savename_list[1]}"
        agent_splitting_type = "femnist"
    elif savename_list[0] == "WBC":
        data_name = f"{savename_list[0]}_{savename_list[1]}"
        agent_splitting_type = f"{savename_list[0]}"
    elif savename_list[0] == "cifar10":
        data_name = f"{savename_list[0]}"
        agent_splitting_type = f"{savename_list[0]}"        

    if data_name == "MNIST_partial":
        # classes = [1, 4, 7]
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        alt_classes = None
        cifar10_noise_type = None
    elif data_name == "FEMNIST_partial":
        # classes = [10, 11, 12] # ABC
        # alt_classes = [36, 37, 38] # abc
        classes = [i for i in range(10, 20)]
        alt_classes = [i for i in range(36, 46)]
        cifar10_noise_type = None
    elif (data_name == "retina") or (data_name == "EyePACS") or (data_name == "EyePACS_wtest"):
        classes = [0, 1, 2, 3, 4]
        alt_classes = None
        cifar10_noise_type = None
    elif data_name == "retinalOCT":
        classes = [0, 1, 2, 3]
        alt_classes = None
        cifar10_noise_type = None
    elif data_name == "retinalOCT_partial":
        classes = [0, 3]
        alt_classes = None
        cifar10_noise_type = None
    elif data_name == "WBC_partial":
        classes = [0, 1, 4, 5]
        alt_classes = None
        cifar10_noise_type = None
    elif data_name == "cifar10":
        # classes = [3, 4, 5]
        # classes = [0, 1, 8, 9]
        # classes = [3, 4, 5]
        classes = [i for i in range(10)]
        alt_classes = None
        cifar10_noise_type = f"{savename_list[1]}"

    # include = ["baseline", "baseline_COD", "oracle", "oracleth", "oracle_COD", "oracle_CODth",
    #            "oracle_CODth_picky", "proposed", "proposed_picky"]
    # include = ["baseline", "oracle", "oracleth", "oracle_COD", "oracle_CODth",
    #            "oracle_CODth_picky", "proposed", "proposed_picky"]
    # include = ["baseline", "oracleth", "oracle_CODth_picky", "proposed_picky"]
    # include = ["baseline", "oracleth", "oracle_CODth", "oracle_CODth_picky", "proposed", "proposed_picky"]
    # include = ["baseline", "oracleth", "oracle_CODth", "proposed"]
    include = ["proposed_budget"]

    # proposed_idx_include = [0, 1, 2, 3]
    proposed_idx_include = [0, 3]

    K = 50+1 # K - 1 other agents
    T = 2
    ell0 = 300
    n = 250
    mr0 = 50
    mr1 = 50

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
    # pi_choice_type = "fixed"
    # pi_param = 0
    # K0 = (K-1)
    # pi_choice_type = "beta"
    # pi_param = 0.2
    # K0 = int((K-1)/2)
    pi_choice_type = "uniform"
    pi_param = -1
    K0 = -1
    # pi_choice_type = "fixed"
    # pi_param = 0.9
    # K0 = 15

    ### Hyperparameters ###
    score_name = savename_list[-3]
    if (score_name == "AdaDetect") or (score_name == "labelAdaDetect"):
        if savename_list[-2] == "LR":
            AdaDetect_classifier = "LogisticRegression"
            use_PCA = False
            n_components = None
        elif savename_list[-2] == "SVC":
            AdaDetect_classifier = "SVC"
            use_PCA = False
            n_components = None
        elif savename_list[-2] == "PCALR":
            AdaDetect_classifier = "LogisticRegression"
            use_PCA = True
            n_components = 0.8
        elif savename_list[-2] == "PCASVC":
            AdaDetect_classifier = "SVC"
            use_PCA = True
            n_components = 0.8
        AdaDetect_individual = False

    elif score_name == "OCSVM":
        assert savename_list[-2] == "", ""

    if savename_list[-1] == "SVC":
        model_name = "SVC"
        model_kwargs = {"kernel": "rbf", "gamma": "auto"}
    elif savename_list[-1] == "LR":
        model_name = "LogisticRegression"
        model_kwargs = dict()
    elif savename_list[-1] == "MLP":
        model_name = "MLPClassifier"
        model_kwargs = dict()
    elif savename_list[-1] == "GBC":
        model_name = "GradientBoostingClassifier"
        model_kwargs = dict()
    elif savename_list[-1] == "PCALR":
        model_name = "PCA_LR"
        model_kwargs = {"n_components": 0.8}
    elif savename_list[-1] == "PCASVC":
        model_name = "PCA_SVC"
        model_kwargs = {"n_components": 0.8}
    elif savename_list[-1] == "ResNet18":
        model_name = "ResNet18"
        model_kwargs = {"num_classes": len(classes)}
    elif savename_list[-1] == "ResNet9":
        model_name = "ResNet9"
        model_kwargs = {"num_classes": len(classes)}
    elif savename_list[-1] == "ResNet4":
        model_name = "ResNet4"
        model_kwargs = {"num_classes": len(classes)}
    elif savename_list[-1] == "CNN":
        model_name = "CNN"
        model_kwargs = {"num_classes": len(classes)}
    # model_name = "KNeighborsClassifier"
    # model_kwargs = dict()

    fixed_budget = np.arange(0, K, step=5)

    # num_rejection_thresholds = 11
    # significance_levels = np.linspace(0, 1, num_rejection_thresholds)
    # significance_levels = np.hstack((np.array([0]), np.logspace(-5, 0, num_rejection_thresholds-1)))
    # significance_levels = [0.05, 0.2, 0.5]
    # significance_levels = [0.05, 0.2, 0.5, 0.7]
    # significance_levels = [0.1, 0.5, 0.9]
    significance_levels = [0.01, 0.1]
    # significance_levels = [0.05, 0.5]
    num_rejection_thresholds = len(significance_levels)

    use_Storey = True
    gamma = 0.5 # Storey's hyperparameter conformal contamination test

    pi_th_arr = np.linspace(0, 0.5, num=6) # Threshold conformal contamination test
    # pi_th_arr = np.linspace(0, 0.7, num=8) # Threshold conformal contamination test
    # pi_th_arr = np.linspace(0, 1, num=11) # Threshold conformal contamination test
    len_pi_th_arr = len(pi_th_arr)

    # pi_th_arr_oracle = np.linspace(0.1, 0.9, num=9) # Threshold conformal contamination test
    # pi_th_arr_oracle = np.linspace(0.05, 0.95, num=19) # Threshold conformal contamination test
    pi_th_arr_oracle = np.linspace(0, 1, num=21) # Threshold conformal contamination test
    # pi_th_arr_oracle = np.linspace(0, 1, num=11) # Threshold conformal contamination test
    len_pi_th_arr_oracle = len(pi_th_arr_oracle)

    # chosen_pi_th_idx = 0
    beta = -1 # Significance level conformal outlier detection
    zeta = -1 # Storey's hyperparameter conformal outlier detection

    lambda_ = n//8 / (n+1)
    i0_param = 3
    i0_arr = [mr0//i0_param for pi_th in pi_th_arr]

    # savename = "temp"

    with open(f"ProposedAccuracy/{savename}.txt", "w") as file:
        file.write(f"sims: {sims:d}\n")
        file.write(f"seed: {seed:d}\n")
        file.write(f"K: {K-1:d}\n")
        file.write(f"ell: {ell0:d}\n")
        file.write(f"n: {n:d}\n")
        file.write(f"mr0: {mr0:d}  ;  mr1: {mr1:d}\n")
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
            if AdaDetect_classifier[:3] == "PCA":
                file.write(f"Score PCA components: {n_components:.2f}\n")
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
        if data_name == "cifar10":
            file.write(f"cifar10 noise type: {cifar10_noise_type}\n")
        file.write(f"Use Storey: {use_Storey}\n")
        if model_name[:3] == "PCA":
            file.write(f"Classifier PCA components: {model_kwargs['n_components']:.2f}\n")
        for inc in include:
            file.write(f"{inc}, ")
        file.write("\n")


    ### Class preparation ###
    data_handler = DataHandler(data_name, feature_noise=feature_noise, feature_noise_type=feature_noise_type,
                               classes=classes, alt_classes=alt_classes, bin_p=bin_p, repeats=repeats, block_size=block_size,
                               std=std, distance_cov=distance_cov, cifar10_noise_type=cifar10_noise_type)
    score_handler = ConformalScore(score_name, labels=classes, classifier=AdaDetect_classifier,
                                   use_PCA=use_PCA, n_components=n_components)
    test_handler = ConformalContaminationTest()
    model_handler = SupervisedMachineLearning(model_name, **model_kwargs)

    ### Organize test data ###
    test_data, test_labels = data_handler.return_test_data()

    ### Loop begin ###
    pi_true = np.zeros((sims, K-1))
    pcon_total = np.zeros((sims, len(pi_th_arr), K-1, 4), dtype=np.float32)
    CCTest_data_proposed = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4, 2), dtype=np.float32)

    if "baseline" in include:
        model_scores_baselines = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)
        bought_data_baselines = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)
        used_data_baselines = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)

    if "baseline_COD" in include:
        model_scores_baselines_COD = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)
        bought_data_baselines_COD = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)
        used_data_baselines_COD = np.zeros((sims, len(fixed_budget)+1), dtype=np.float32)
        CODTest_data_baselines_COD = np.zeros((sims, len(fixed_budget), 2), dtype=np.float32)

    # if "DataCV_COD" in include:
    #     model_scores_DataCV_COD = np.zeros((sims, K+1), dtype=np.float32)
    #     bought_data_DataCV_COD = np.zeros((sims, K+1), dtype=np.float32)
    #     used_data_DataCV_COD = np.zeros((sims, K+1), dtype=np.float32)
    #     CODTest_data_DataCV_COD = np.zeros((sims, K, 2), dtype=np.float32)

    if "oracle" in include:
        model_scores_oracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)
        bought_data_oracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)
        used_data_oracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)

    if "oracleth" in include:
        model_scores_oraclesth = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        bought_data_oraclesth = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        used_data_oraclesth = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)

    if "oracle_COD" in include:
        model_scores_partoracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)
        bought_data_partoracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)
        used_data_partoracles = np.zeros((sims, len(fixed_budget)), dtype=np.float32)
        CODTest_data_partoracles = np.zeros((sims, len(fixed_budget), 2), dtype=np.float32)

    if "oracle_CODth" in include:
        model_scores_partoracles_th = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        bought_data_partoracles_th = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        used_data_partoracles_th = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        CODTest_data_partoracles_th = np.zeros((sims, len_pi_th_arr_oracle, 2), dtype=np.float32)

    if "oracle_CODth_picky" in include:
        model_scores_partoracles_th_picky = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        bought_data_partoracles_th_picky = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        used_data_partoracles_th_picky = np.zeros((sims, len_pi_th_arr_oracle), dtype=np.float32)
        CODTest_data_partoracles_th_picky = np.zeros((sims, len_pi_th_arr_oracle, 2), dtype=np.float32)

    if "proposed" in include:
        model_scores_proposed = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        bought_data_proposed = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        used_data_proposed = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        CODTest_data_proposed = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4, 2), dtype=np.float32)

    if "proposed_budget" in include:
        model_scores_proposed_budget = np.zeros((sims, len(fixed_budget), 4), dtype=np.float32)
        bought_data_proposed_budget = np.zeros((sims, len(fixed_budget), 4), dtype=np.float32)
        used_data_proposed_budget = np.zeros((sims, len(fixed_budget), 4), dtype=np.float32)
        CODTest_data_proposed_budget = np.zeros((sims, len(fixed_budget), 4, 2), dtype=np.float32)

    if "proposed_picky" in include:
        model_scores_proposed_picky = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        bought_data_proposed_picky = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        used_data_proposed_picky = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
        CODTest_data_proposed_picky = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4, 2), dtype=np.float32)

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
                                                pi_k=pi_k, randomize=randomize, mr0=mr0, mr1=mr1, T=T, bin_p=bin_p, repeats=repeats, block_size=block_size,
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
                                                                            AdaDetect_individual, mr0, mr1, K-1)
        allin_conformal_pvalues = test_handler.compute_conformal_pvalues(calibration_scores, allin_data_scores)

        if "baseline" in include:
            ### Baseline: no data sharing ###
            model_handler.fit(train_data, train_labels) # SLOW
            model_scores_baselines[sim, 0] = model_handler.score(test_data, test_labels)
            bought_data_baselines[sim, 0] = 0
            used_data_baselines[sim, 0] = train_data.shape[0]

            ### Baseline : rho-fixed data sharing ###
            for rho_idx, rho in enumerate(fixed_budget):
                print("\t Random:", rho, "/", K-1)
                additional_train_data, additional_train_labels \
                    = ut.import_data_time2_fixed(data_dict, rho, K, potential_train_data, potential_train_labels)

                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))
                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_baselines[sim, rho_idx+1] = model_handler.score(test_data, test_labels)
                bought_data_baselines[sim, rho_idx+1] = mr0*(K-1) + mr1*rho
                used_data_baselines[sim, rho_idx+1] = temp_train_data.shape[0]

        if "baseline_COD" in include:
            ## Baseline : rho-fixed data sharing with conformal outlier detection ###
            for rho_idx, rho in enumerate(fixed_budget):
                print("\t Random (COD):", rho, "/", K-1)
                temp_indicator_arr = ut.choose_data_subset_fixed(rho, K, mr0, mr1)
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

                FDP, TDP = ut.compute_FDP_TDP(rejectBoolCOD, temp_null_indicator)

                additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))

                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_baselines_COD[sim, rho_idx] = model_handler.score(test_data, test_labels)
                bought_data_baselines_COD[sim, rho_idx] = mr0*(K-1) + mr1*rho
                used_data_baselines_COD[sim, rho_idx] = temp_train_data.shape[0]
                CODTest_data_baselines_COD[sim, rho_idx, 0] = FDP
                CODTest_data_baselines_COD[sim, rho_idx, 1] = TDP

        if "DataCV_COD" in include:
            pass

        if "oracle" in include:
            ### Baseline: oracle budget based ###
            sort_idx = np.argsort(pi_k)
            for rho_idx, rho in enumerate(fixed_budget):
                print("\t Oracle:", rho, "/", K-1)
                additional_train_data, additional_train_labels, additional_train_null_indicator \
                    = ut.import_data_time2_oracle(data_dict, rho, K, potential_train_data,
                                                  potential_train_labels, potential_train_null_indicator, sort_idx)

                additional_train_data = additional_train_data[np.invert(additional_train_null_indicator)]
                additional_train_labels = additional_train_labels[np.invert(additional_train_null_indicator)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))
                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_oracles[sim, rho_idx] = model_handler.score(test_data, test_labels)
                bought_data_oracles[sim, rho_idx] = mr0*(K-1) + mr1*rho
                used_data_oracles[sim, rho_idx] = temp_train_data.shape[0]

        if "oracleth" in include:
            ### Baseline: oracle threshold based ###
            for pi_th_idx, pi_th in enumerate(pi_th_arr_oracle):
                print("\t Oracle threshold:", pi_th_idx+1, "/", len_pi_th_arr_oracle)
                additional_train_data, additional_train_labels, additional_train_null_indicator \
                    = ut.import_data_time2_oracleth(data_dict, pi_th, pi_k)

                additional_train_data = additional_train_data[np.invert(additional_train_null_indicator)]
                additional_train_labels = additional_train_labels[np.invert(additional_train_null_indicator)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))
                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_oraclesth[sim, pi_th_idx] = model_handler.score(test_data, test_labels)
                bought_data_oraclesth[sim, pi_th_idx] = mr0*(K-1) + mr1*np.sum(pi_k <= pi_th)
                used_data_oraclesth[sim, pi_th_idx] = temp_train_data.shape[0]

        if "oracle_COD" in include:
            ### Baseline: oracle with conformal outlier detection ###
            sort_idx = np.argsort(pi_k)
            for rho_idx, rho in enumerate(fixed_budget):
                print("\t Oracle all data:", rho, "/", K-1)
                temp_indicator_arr = ut.choose_data_subset_oracle(rho, K, mr0, mr1, sort_idx)
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

                FDP, TDP = ut.compute_FDP_TDP(rejectBoolCOD, temp_null_indicator)

                additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))

                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_partoracles[sim, rho_idx] = model_handler.score(test_data, test_labels)
                bought_data_partoracles[sim, rho_idx] = mr0*(K-1) + mr1*rho
                used_data_partoracles[sim, rho_idx] = temp_train_data.shape[0]
                CODTest_data_partoracles[sim, rho_idx, 0] = FDP
                CODTest_data_partoracles[sim, rho_idx, 1] = TDP

        if "oracle_CODth" in include:
            ### Baseline: oracle with contamination threshold and conformal outlier detection ###
            for pi_th_idx, pi_th in enumerate(pi_th_arr_oracle):
                print("\t Oracle threshold all data:", pi_th_idx+1, "/", len_pi_th_arr_oracle)
                temp_indicator_arr = ut.choose_data_subset_oracle_th(pi_th, mr0, mr1, pi_k)
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

                FDP, TDP = ut.compute_FDP_TDP(rejectBoolCOD, temp_null_indicator)

                additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))

                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_partoracles_th[sim, pi_th_idx] = model_handler.score(test_data, test_labels)
                bought_data_partoracles_th[sim, pi_th_idx] = mr0*(K-1) + mr1*np.sum(pi_k <= pi_th)
                used_data_partoracles_th[sim, pi_th_idx] = temp_train_data.shape[0]
                CODTest_data_partoracles_th[sim, pi_th_idx, 0] = FDP
                CODTest_data_partoracles_th[sim, pi_th_idx, 1] = TDP

        if "oracle_CODth_picky" in include:
            ### Baseline: oracle with contamination threshold and conformal outlier detection ###
            for pi_th_idx, pi_th in enumerate(pi_th_arr_oracle):
                print("\t Oracle threshold picky:", pi_th_idx+1, "/", len_pi_th_arr_oracle)
                temp_indicator_arr = ut.choose_data_subset_oracle_th_picky(pi_th, mr0, mr1, pi_k)
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

                FDP, TDP = ut.compute_FDP_TDP(rejectBoolCOD, temp_null_indicator)

                additional_train_data = temp_potential_train_data[np.invert(rejectBoolCOD)]
                additional_train_labels = temp_potential_train_labels[np.invert(rejectBoolCOD)]
                temp_train_data = np.concatenate((train_data, additional_train_data), axis=0)
                temp_train_labels = np.hstack((train_labels, additional_train_labels))

                model_handler.fit(temp_train_data, temp_train_labels) # SLOW
                model_scores_partoracles_th_picky[sim, pi_th_idx] = model_handler.score(test_data, test_labels)
                bought_data_partoracles_th_picky[sim, pi_th_idx] = mr0*(K-1) + mr1*np.sum(pi_k <= pi_th)
                used_data_partoracles_th_picky[sim, pi_th_idx] = temp_train_data.shape[0]
                CODTest_data_partoracles_th_picky[sim, pi_th_idx, 0] = FDP
                CODTest_data_partoracles_th_picky[sim, pi_th_idx, 1] = TDP

        ### Run conformal contamination test ###
        test_scores = np.zeros((K-1, mr0), dtype=np.float64)
        for k in range(1, K):
            # test_scores[k-1] = score_handler.score(data_dict[f"Agent{k}_Time{0}_test"][0], data_dict[f"Agent{k}_Time{0}_test"][1])
            test_scores[k-1] = allin_data_scores[(k-1)*mr0:k*mr0]

        pcon = np.zeros((len(pi_th_arr), K-1, 4), dtype=np.float32)
        for pi_th_idx, pi_th in enumerate(pi_th_arr):
            for k in range(1, K):
                pcon[pi_th_idx, k-1, 0], pcon[pi_th_idx, k-1, 1], pcon[pi_th_idx, k-1, 2], pcon[pi_th_idx, k-1, 3] \
                    = test_handler.all_conformal_contamination_tests(calibration_scores, test_scores[k-1],
                      pi_th=pi_th, n=n, m=mr0, lambda_=lambda_, i0=i0_arr[pi_th_idx])
                pcon_total[sim] = pcon
                # print(pcon)

        rejectBoolCCTest = np.zeros((len(pi_th_arr), num_rejection_thresholds, 4, K-1), dtype=bool)
        for idx_alpha, alpha in enumerate(significance_levels):
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for idx in proposed_idx_include:
                    pvals = pcon[pi_th_idx, :, idx]
                    if use_Storey is True:
                        _, K0_hat = Storeys_correction(gamma, pvals, K-1)
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)
                    elif use_Storey is False:
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K-1, K-1)
                    rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :] = rejectBool_

                    # FDP
                    R = np.sum(rejectBool_)
                    RcapH = np.sum(rejectBool_[pi_k<=pi_th])
                    if R == 0:
                        FDP = 0
                    else:
                        FDP = RcapH/R
                    CCTest_data_proposed[sim, pi_th_idx, idx_alpha, idx, 0] = FDP

                    # TDP
                    K1P = np.sum(pi_k>pi_th)
                    RcapHc = np.sum(rejectBool_[pi_k>pi_th])
                    if K1P == 0:
                        TDP = 0
                    else:
                        TDP = RcapHc/K1P
                    CCTest_data_proposed[sim, pi_th_idx, idx_alpha, idx, 1] = TDP
        CCTest_data_proposed[sim, :, 0, :, 0] = 0
        CCTest_data_proposed[sim, :, 0, :, 1] = 0
        CCTest_data_proposed[sim, :, -1, :, 0] = 1
        CCTest_data_proposed[sim, :, -1, :, 1] = 1

        if "proposed" in include:
            for idx_alpha, alpha in enumerate(significance_levels):
                for pi_th_idx, pi_th in enumerate(pi_th_arr):
                    print(f"\t Proposed: ({idx_alpha+1}, {pi_th_idx+1}) / ({num_rejection_thresholds}, {len_pi_th_arr})")
                    for idx in proposed_idx_include:
                        bought_data_proposed[sim, pi_th_idx, idx_alpha, idx] = mr0*(K-1) + mr1*((K-1) - np.sum(rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :]))
                        model_scores_proposed[sim, pi_th_idx, idx_alpha, idx], used_data_proposed[sim, pi_th_idx, idx_alpha, idx], \
                        CODTest_data_proposed[sim, pi_th_idx, idx_alpha, idx, 0], CODTest_data_proposed[sim, pi_th_idx, idx_alpha, idx, 1] \
                            = ut.proposed(rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :], K, mr0, mr1, allin_conformal_pvalues,
                                          allin_train_data, allin_train_labels, train_data, train_labels,
                                          zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels)

        if "proposed_budget" in include:
            for rho_idx, rho in enumerate(fixed_budget):
                print("\t Proposed budget:", rho, "/", K-1)
                for idx in proposed_idx_include:
                    rejectBool_ = np.zeros(K-1, dtype=bool)
                    stat = pcon[0, :, idx]
                    stat_argsort = np.argsort(stat)
                    rejectBool_[stat_argsort[:rho]] = 1
                    bought_data_proposed_budget[sim, rho_idx, idx] = mr0*(K-1) + mr1*rho
                    model_scores_proposed_budget[sim, rho_idx, idx], used_data_proposed_budget[sim, rho_idx, idx], \
                    CODTest_data_proposed_budget[sim, rho_idx, idx, 0], CODTest_data_proposed_budget[sim, rho_idx, idx, 1] \
                        = ut.proposed(rejectBool_, K, mr0, mr1, allin_conformal_pvalues,
                                      allin_train_data, allin_train_labels, train_data, train_labels,
                                      zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels)

        if "proposed_picky" in include:
            for idx_alpha, alpha in enumerate(significance_levels):
                for pi_th_idx, pi_th in enumerate(pi_th_arr):
                    print(f"\t Proposed picky: ({idx_alpha+1}, {pi_th_idx+1}) / ({num_rejection_thresholds}, {len_pi_th_arr})")
                    for idx in proposed_idx_include:
                        bought_data_proposed_picky[sim, pi_th_idx, idx_alpha, idx] = mr0*(K-1) + mr1*((K-1) - np.sum(rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :]))
                        model_scores_proposed_picky[sim, pi_th_idx, idx_alpha, idx], used_data_proposed_picky[sim, pi_th_idx, idx_alpha, idx], \
                        CODTest_data_proposed_picky[sim, pi_th_idx, idx_alpha, idx, 0], CODTest_data_proposed_picky[sim, pi_th_idx, idx_alpha, idx, 1] \
                            = ut.proposed_picky(rejectBoolCCTest[pi_th_idx, idx_alpha, idx, :], K, mr0, mr1, allin_conformal_pvalues,
                                                allin_train_data, allin_train_labels, train_data, train_labels,
                                                zeta, beta, model_handler, allin_train_null_indicator, test_data, test_labels)

    kwds = {"pi_true": pi_true,
            "pcon_total": pcon_total,
            "CCTest_data_proposed": CCTest_data_proposed}

    if "baseline" in include:
        kwds.update({"model_scores_baselines": model_scores_baselines,
                     "bought_data_baselines": bought_data_baselines,
                     "used_data_baselines": used_data_baselines})

    if "baseline_COD" in include:
        kwds.update({"model_scores_baselines_COD": model_scores_baselines_COD,
                     "bought_data_baselines_COD": bought_data_baselines_COD,
                     "used_data_baselines_COD": used_data_baselines_COD,
                     "CODTest_data_baselines_COD": CODTest_data_baselines_COD})

    # if "DataCV_COD" in include:
    #     kwds.update({"model_scores_DataCV_COD": model_scores_DataCV_COD,
    #                  "bought_data_DataCV_COD": bought_data_DataCV_COD,
    #                  "used_data_DataCV_COD": used_data_DataCV_COD,
    #                  "CODTest_data_DataCV_COD": CODTest_data_DataCV_COD})

    if "oracle" in include:
        kwds.update({"model_scores_oracles": model_scores_oracles,
                     "bought_data_oracles": bought_data_oracles,
                     "used_data_oracles": used_data_oracles})

    if "oracleth" in include:
        kwds.update({"model_scores_oraclesth": model_scores_oraclesth,
                     "bought_data_oraclesth": bought_data_oraclesth,
                     "used_data_oraclesth": used_data_oraclesth})

    if "oracle_COD" in include:
        kwds.update({"model_scores_partoracles": model_scores_partoracles,
                     "bought_data_partoracles": bought_data_partoracles,
                     "used_data_partoracles": used_data_partoracles,
                     "CODTest_data_partoracles": CODTest_data_partoracles})

    if "oracle_CODth" in include:
        kwds.update({"model_scores_partoracles_th": model_scores_partoracles_th,
                     "bought_data_partoracles_th": bought_data_partoracles_th,
                     "used_data_partoracles_th": used_data_partoracles_th,
                     "CODTest_data_partoracles_th": CODTest_data_partoracles_th})

    if "oracle_CODth_picky" in include:
        kwds.update({"model_scores_partoracles_th_picky": model_scores_partoracles_th_picky,
                     "bought_data_partoracles_th_picky": bought_data_partoracles_th_picky,
                     "used_data_partoracles_th_picky": used_data_partoracles_th_picky,
                     "CODTest_data_partoracles_th_picky": CODTest_data_partoracles_th_picky})

    if "proposed" in include:
        kwds.update({"model_scores_proposed": model_scores_proposed,
                     "bought_data_proposed": bought_data_proposed,
                     "used_data_proposed": used_data_proposed,
                     "CODTest_data_proposed": CODTest_data_proposed})

    if "proposed budget" in include:
        kwds.update({"model_scores_proposed_budget": model_scores_proposed_budget,
                     "bought_data_proposed_budget": bought_data_proposed_budget,
                     "used_data_proposed_budget": used_data_proposed_budget,
                     "CODTest_data_proposed_budget": CODTest_data_proposed_budget})

    if "proposed_picky" in include:
        kwds.update({"model_scores_proposed_picky": model_scores_proposed_picky,
                     "bought_data_proposed_picky": bought_data_proposed_picky,
                     "used_data_proposed_picky": used_data_proposed_picky,
                     "CODTest_data_proposed_picky": CODTest_data_proposed_picky})

    np.savez(f"ProposedAccuracy/{savename}.npz", **kwds)

    # np.savez(f"ProposedAccuracy/{savename}.npz",
    #          model_scores_baselines = model_scores_baselines, bought_data_baselines = bought_data_baselines, used_data_baselines = used_data_baselines,
    #          model_scores_baselines_COD = model_scores_baselines_COD, bought_data_baselines_COD = bought_data_baselines_COD, used_data_baselines_COD = used_data_baselines_COD,
    #          CODTest_data_baselines_COD = CODTest_data_baselines_COD,
    #          model_scores_oracles = model_scores_oracles, bought_data_oracles = bought_data_oracles, used_data_oracles = used_data_oracles,
    #          model_scores_oraclesth = model_scores_oraclesth, bought_data_oraclesth = bought_data_oraclesth, used_data_oraclesth = used_data_oraclesth,
    #          model_scores_partoracles = model_scores_partoracles, bought_data_partoracles = bought_data_partoracles, used_data_partoracles = used_data_partoracles,
    #          CODTest_data_partoracles = CODTest_data_partoracles,
    #          model_scores_partoracles_th = model_scores_partoracles_th, bought_data_partoracles_th = bought_data_partoracles_th, used_data_partoracles_th = used_data_partoracles_th,
    #          CODTest_data_partoracles_th = CODTest_data_partoracles_th,
    #          model_scores_partoracles_th_picky = model_scores_partoracles_th_picky, bought_data_partoracles_th_picky = bought_data_partoracles_th_picky, used_data_partoracles_th_picky = used_data_partoracles_th_picky,
    #          CODTest_data_partoracles_th_picky = CODTest_data_partoracles_th_picky,
    #          model_scores_proposed = model_scores_proposed, bought_data_proposed = bought_data_proposed, used_data_proposed = used_data_proposed,
    #          pcon_total = pcon_total, CCTest_data_proposed = CCTest_data_proposed, CODTest_data_proposed = CODTest_data_proposed,
    #          model_scores_proposed_picky = model_scores_proposed_picky, bought_data_proposed_picky = bought_data_proposed_picky, used_data_proposed_picky = used_data_proposed_picky,
    #          CODTest_data_proposed_picky = CODTest_data_proposed_picky,
    #          pi_true = pi_true)

    if "baseline" in include:
        model_scores_baselines_mean = np.mean(model_scores_baselines, axis=0)
        bought_data_baselines_mean = np.mean(bought_data_baselines, axis=0)

    if "baseline_COD" in include:
        model_scores_baselines_COD_mean = np.mean(model_scores_baselines_COD, axis=0)
        bought_data_baselines_COD_mean = np.mean(bought_data_baselines_COD, axis=0)

    # if "DataCV_COD" in include:
    #     model_scores_DataCV_COD_mean = np.mean(model_scores_DataCV_COD, axis=0)
    #     bought_data_DataCV_COD_mean = np.mean(bought_data_DataCV_COD, axis=0)

    if "oracle" in include:
        model_scores_oracles_mean = np.mean(model_scores_oracles, axis=0)
        bought_data_oracles_mean = np.mean(bought_data_oracles, axis=0)

    if "oracleth" in include:
        model_scores_oraclesth_mean = np.mean(model_scores_oraclesth, axis=0)
        bought_data_oraclesth_mean = np.mean(bought_data_oraclesth, axis=0)

    if "oracle_COD" in include:
        model_scores_partoracles_mean = np.mean(model_scores_partoracles, axis=0)
        bought_data_partoracles_mean = np.mean(bought_data_partoracles, axis=0)

    if "oracle_CODth" in include:
        model_scores_partoracles_th_mean = np.mean(model_scores_partoracles_th, axis=0)
        bought_data_partoracles_th_mean = np.mean(bought_data_partoracles_th, axis=0)

    if "oracle_CODth_picky" in include:
        model_scores_partoracles_th_picky_mean = np.mean(model_scores_partoracles_th_picky, axis=0)
        bought_data_partoracles_th_picky_mean = np.mean(bought_data_partoracles_th_picky, axis=0)

    if "proposed" in include:
        model_scores_proposed_mean = np.mean(model_scores_proposed, axis=0)
        bought_data_proposed_mean = np.mean(bought_data_proposed, axis=0)

    if "proposed_picky" in include:
        model_scores_proposed_picky_mean = np.mean(model_scores_proposed_picky, axis=0)
        bought_data_proposed_picky_mean = np.mean(bought_data_proposed_picky, axis=0)


    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "k"]
    _ = plt.figure(figsize=fsize)
    if "baseline" in include:
        plt.plot(bought_data_baselines_mean, model_scores_baselines_mean, "o", color=colors[0], label="Fixed budget")
    if "baseline_COD" in include:
        plt.plot(bought_data_baselines_COD_mean, model_scores_baselines_COD_mean, "*", color=colors[0], label="Fixed budget (COD)")
    # if "DataCV_COD" in include:
    #     plt.plot(bought_data_DataCV_COD_mean/m, model_scores_DataCV_COD_mean, "*", color=colors[10], label="Data CV (COD)")
    if "oracle" in include:
        plt.plot(bought_data_oracles_mean, model_scores_oracles_mean, "o", color=colors[2], label="Oracle")
    if "oracleth" in include:
        plt.plot(bought_data_oraclesth_mean, model_scores_oraclesth_mean, "1", color=colors[2], label="Oracle threshold")
    if "oracle_COD" in include:
        plt.plot(bought_data_partoracles_mean, model_scores_partoracles_mean, "*", color=colors[9], label="Oracle all data")
    if "oracle_CODth" in include:
        plt.plot(bought_data_partoracles_th_mean, model_scores_partoracles_th_mean, "1", color=colors[9], label="Oracle threshold all data")
    if "oracle_CODth_picky" in include:
        plt.plot(bought_data_partoracles_th_picky_mean, model_scores_partoracles_th_picky_mean, "x", color=colors[9], label="Oracle picky")
    if "proposed" in include:
        plt.plot(bought_data_proposed_mean[:, :, 0].flatten(), model_scores_proposed_mean[:, :, 0].flatten(), "o", color=colors[1], label="Storey")
        plt.plot(bought_data_proposed_mean[:, :, 1].flatten(), model_scores_proposed_mean[:, :, 1].flatten(), "o", color=colors[8], label="Quantile")
        # plt.plot(bought_data_proposed_mean[pi_th_idx, :, 2], model_scores_proposed_mean[pi_th_idx, :, 2], "o", color=colors[4], label="Summation")
        # plt.plot(bought_data_proposed_mean[pi_th_idx, :, 3], model_scores_proposed_mean[pi_th_idx, :, 3], "o", color=colors[1], label="Fisher")
    if "proposed_picky" in include:
        plt.plot(bought_data_proposed_picky_mean[:, :, 0].flatten(), model_scores_proposed_picky_mean[:, :, 0].flatten(), "x", color=colors[1], label="Storey picky")
        plt.plot(bought_data_proposed_picky_mean[:, :, 1].flatten(), model_scores_proposed_picky_mean[:, :, 1].flatten(), "x", color=colors[8], label="Quantile picky")
    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # print(bought_data_proposed_mean[:, :, 0])
    # print(model_scores_proposed_mean[:, :, 0])


if __name__ == "__main__":
    main()