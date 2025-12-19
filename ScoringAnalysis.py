#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:47:28 2025

In this script the conformal contamination tests are numerically analyzed.
"""

import numpy as np
from time import time

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

from utilities import simulate_contamination_beta, import_data_time1, compute_conformal_scores

from ConformalScoreModule import ConformalScore
from DataHandlerModule import DataHandler
from ConformalContaminationTestModule import ConformalContaminationTest
from Benjamini_Hochberg import Storeys_correction, Benjamini_Hochberg_procedure

def main():
    seed = 24
    np.random.seed(seed)
    sims = 10
    randomize = True
    # savename = "MNIST_partial_ln_labelAdaDetect_PCALR_discrete_uniform"
    savename = "temp"

    import warnings
    warnings.filterwarnings("ignore")

    ### Scenario settings ###
    data_name = "imagenet"
    agent_splitting_type = "imagenet"
    if data_name == "MNIST_partial":
        classes = [1, 4, 7]
        alt_classes = None
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2
        feature_dim = (28, 28, 1)
    elif data_name == "FEMNIST_partial":
        classes = [10, 11, 12] # ABC
        alt_classes = [36, 37, 38] # abc
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2
        feature_dim = (28, 28, 1)
    elif (data_name == "retina") or (data_name == "EyePACS") or (data_name == "EyePACS_wtest"):
        classes = [0, 1, 2, 3, 4]
        alt_classes = None
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2*3
        feature_dim = (28, 28, 3)
    elif data_name == "retinalOCT":
        classes = [0, 1, 2, 3]
        alt_classes = None
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2
        feature_dim = (28, 28, 1)
    elif data_name == "retinalOCT_partial":
        classes = [0, 3]
        alt_classes = None
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2
        feature_dim = (28, 28, 1)
    elif data_name == "WBC_partial":
        classes = [0, 1, 4, 5]
        alt_classes = None
        cifar10_noise_type = None
        imagenet_noise_type = None
        shape = 28**2*3
        feature_dim = (28, 28, 3)
    elif data_name == "cifar10":
        classes = [3, 4, 5]
        # classes = [0, 1, 8, 9]
        # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        alt_classes = None
        shape = 32**2*3
        cifar10_noise_type = "brightness"
        imagenet_noise_type = None
        feature_dim = (32, 32, 3)
    elif data_name == "imagenet":
        classes = [i for i in range(40)]
        alt_classes = None
        shape = 64**2*3
        cifar10_noise_type = None
        imagenet_noise_type = "brightness"
        feature_dim = (64, 64, 3)

    K = 50+1 # K - 1 other agents
    T = 2
    ell0 = len(classes)*20
    n = len(classes)*15
    m = 30
    mr1 = 1

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
        alt_noise_type = "GRG_norm"
    else:
        feature_noise_type = "GRF_norm"
        feature_noise = "train_test"
        alt_noise_type = None

    ### Contamination ###
    pi_choice_type = "discrete_uniform"

    if pi_choice_type == "fixed":
        pi_param = -1
        K0 = int(K-1)
    elif pi_choice_type == "fixed_part":
        pi_param = 0.1
        K0 = int((K-1)/2) # int(K-1)
    elif pi_choice_type == "fixed_balanced":
        K0 = int((K-1)/2)
    else:
        pi_param = 0.2
        K0 = int((K-1)/2)

    ### Hyperparameters ###
    score_name = "AdaDetect"
    # AdaDetect_classifier = "LogisticRegression"
    # AdaDetect_classifier = "SVC"
    # AdaDetect_classifier = "MLPClassifier"
    # AdaDetect_classifier = "CNN"
    AdaDetect_classifier = "ResNet4"

    AdaDetect_individual = False
    use_PCA = False
    n_components = 0.8

    use_Storey = False
    gamma = 0.5 # Storey's hyperparameter conformal contamination test
    num_rejection_thresholds = 40
    rejection_thresholds = np.linspace(0, 0.975, num_rejection_thresholds)
    significance_levels = rejection_thresholds

    # pi_th_arr = np.linspace(0, 0.3, num=7) # Threshold conformal contamination test
    pi_th_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    len_pi_th_arr = len(pi_th_arr)

    lambda_ = n//8 / (n+1)
    i0_param = 3
    i0_arr = [m//i0_param for pi_th in pi_th_arr]

    if score_name == "AutoencoderXY":
        shape += 1
        latent_dim = 64
    elif (score_name == "Autoencoder") or (score_name == "labelAutoencoder"):
        latent_dim = 64
    else:
        latent_dim = -1

    with open(f"ScoringResults/{savename}.txt", "w") as file:
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
        if pi_choice_type == "fixed_part" or pi_choice_type == "beta":
            file.write(f"pi parameter: {pi_param:.2f}\n")
            file.write(f"K0: {K0:d}\n")
        file.write(f"gamma: {gamma:.2f}\n")
        file.write(f"lambda: {lambda_:.2f}\n")    
        file.write(f"i0 parameter: {i0_param:.2f}\n")
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
        if data_name == "cifar10":
            file.write(f"cifar10 noise type: {cifar10_noise_type}\n")
        if data_name == "imagenet":
            file.write(f"imagenet noise type: {imagenet_noise_type}\n")
        file.write(f"Use Storey: {use_Storey}\n")
        if (score_name == "AutoencoderXY") or (score_name == "Autoencoder") or (score_name == "labelAutoencoder"):
            file.write(f"Autoencoder latent dim: {latent_dim:d}\n")
        file.write("\n")
        if (use_PCA is True):
            file.write("use PCA: True\n")
            file.write(f"n components: {n_components:.2f}\n")
        file.write("\n")

    ### Class preparation ###
    data_handler = DataHandler(data_name, feature_noise=feature_noise, feature_noise_type=feature_noise_type,
                               classes=classes, alt_classes=alt_classes, bin_p=bin_p, repeats=repeats, block_size=block_size,
                               std=std, distance_cov=distance_cov, cifar10_noise_type=cifar10_noise_type, imagenet_noise_type=imagenet_noise_type)
    score_handler = ConformalScore(score_name, labels=classes, classifier=AdaDetect_classifier,
                                   shape=shape, latent_dim=latent_dim, use_PCA=use_PCA, n_components=n_components,
                                   feature_dim = feature_dim)
    test_handler = ConformalContaminationTest()


    ### Loop begin ###
    pi_true = np.zeros((sims, len_pi_th_arr, K-1))
    pcon_total = np.zeros((sims, len_pi_th_arr, K-1, 4), dtype=np.float32)
    # FDR_est = np.ones((sims, len(pi_th_arr), K, 4), dtype=np.float32)
    # FDR_est_v2 = np.ones((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.float32)
    # CCTest_data_proposed = np.zeros((sims, len(pi_th_arr), K, 4, 2), dtype=np.float32)
    # CCTest_data_proposed_v2 = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4, 2), dtype=np.float32)
    # bought_v2 = np.zeros((sims, len(pi_th_arr), num_rejection_thresholds, 4), dtype=np.uint16)
    CCTest_data_proposed_v3 = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4, 2), dtype=np.float32)
    bought_v3 = np.zeros((sims, len_pi_th_arr, num_rejection_thresholds, 4), dtype=np.uint16)
    # pvals_all = np.zeros((sims, len_pi_th_arr, m, K-1), dtype=np.float64)
    pvals_all = np.zeros(0, dtype=np.float32)
    indicator_all = np.zeros(0, dtype=bool)
    labels_all = np.zeros(0, dtype=np.int16)
    tot_time_scores = 0
    tot_time_contamination = 0
    for sim in range(sims):
        print("Simulation:", sim+1, "/", sims)
        if (pi_choice_type == "fixed") or (pi_choice_type == "fixed_part") or (pi_choice_type == "fixed_balanced"):
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                if pi_choice_type == "fixed":
                    assert pi_th <= 1, ""
                    pi_k = pi_th * np.ones(K-1)
                elif pi_choice_type == "fixed_part":
                    assert pi_th + pi_param <= 1, ""
                    pi_null = pi_th * np.ones(K0)
                    pi_alt = pi_th + pi_param * np.ones(K-1-K0)
                    randomize_array = np.random.permutation(K-1)
                    pi_k = np.hstack((pi_null, pi_alt))[randomize_array]
                elif pi_choice_type == "fixed_balanced":
                    assert pi_th <= 0.5, ""
                    pi_null = pi_th * np.ones(K0)
                    pi_alt = 1 - pi_th * np.ones(K-1-K0)
                    randomize_array = np.random.permutation(K-1)
                    pi_k = np.hstack((pi_null, pi_alt))[randomize_array]
                pi_true[sim, pi_th_idx, :] = pi_k
                print(pi_k)

                data_dict = data_handler.data_splitting(K=K, K0=None, agent_splitting_type=agent_splitting_type, ell0=ell0, n0=n,
                                                        pi_k=pi_k, randomize=randomize, mr0=m, mr1=mr1, T=T, bin_p=bin_p, repeats=repeats, block_size=block_size,
                                                        std=std, distance_cov=distance_cov, std_alt=std_alt, distance_cov_alt=distance_cov_alt, alt_noise_type=alt_noise_type, bin_p_alt=bin_p_alt)

                ### Organize data (part 1) ###
                train_data = data_dict["Agent0_train"][0]
                train_data = np.concatenate((train_data, data_dict["Agent0_calibration"][0]), axis=0)
                train_labels = data_dict["Agent0_train"][1]
                train_labels = np.hstack((train_labels, data_dict["Agent0_calibration"][1]))

                potential_train_data, potential_train_labels, potential_train_null_indicator = import_data_time1(data_dict, data_handler.d, K)

                ### All the data ###
                # allin_train_data, allin_train_labels, allin_train_null_indicator \
                #     = import_data_time2_all(data_dict, K, potential_train_data, potential_train_labels, potential_train_null_indicator)

                ### Conformal scores and p-values ###
                t1 = time()
                calibration_scores, potential_data_scores = compute_conformal_scores(score_name, score_handler, data_dict,
                                                                                     potential_train_data, potential_train_labels,
                                                                                     AdaDetect_individual, m, mr1, K-1)
                potential_conformal_pvalues = test_handler.compute_conformal_pvalues(calibration_scores, potential_data_scores)
                t2 = time()
                tot_time_scores += t2-t1

                # pvals_all[sim, pi_th_idx] = potential_conformal_pvalues.reshape((m, K-1))
                pvals_all = np.hstack((pvals_all, potential_conformal_pvalues))
                indicator_all = np.hstack((indicator_all, potential_train_null_indicator))
                labels_all = np.hstack((labels_all, potential_train_labels))

                ### Run conformal contamination test ###
                test_scores = np.zeros((K-1, m), dtype=np.float64)
                for k in range(1, K):
                    # test_scores[k-1] = score_handler.score(data_dict[f"Agent{k}_Time{0}_test"][0], data_dict[f"Agent{k}_Time{0}_test"][1])
                    test_scores[k-1] = potential_data_scores[(k-1)*m:k*m]

                t1 = time()
                pcon = np.zeros((K-1, 4), dtype=np.float32)
                # for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for k in range(1, K):
                    pcon[k-1, 0], pcon[k-1, 1], pcon[k-1, 2], pcon[k-1, 3] \
                        = test_handler.all_conformal_contamination_tests(calibration_scores, test_scores[k-1],
                          pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0_arr[pi_th_idx])
                    pcon_total[sim, pi_th_idx] = pcon
                t2 = time()
                tot_time_contamination += t2-t1

                ### Storey's Benjamini-Hochberg
                for idx_alpha, alpha in enumerate(significance_levels):
                    # for pi_th_idx, pi_th in enumerate(pi_th_arr):
                    for idx in range(4):
                        pvals = pcon[:, idx]
                        if use_Storey is True:
                            _, K0_hat = Storeys_correction(gamma, pvals, K-1)
                            rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)
                        elif use_Storey is False:
                            rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K-1, K-1)

                        # FDP
                        R = np.sum(rejectBool_)
                        RcapH = np.sum(rejectBool_[pi_k<=pi_th])
                        if R == 0:
                            FDP = 0
                        else:
                            FDP = RcapH/R
                        CCTest_data_proposed_v3[sim, pi_th_idx, idx_alpha, idx, 0] = FDP

                        # TDP
                        K1P = np.sum(pi_k>pi_th)
                        RcapHc = np.sum(rejectBool_[pi_k>pi_th])
                        if K1P == 0:
                            TDP = 0
                        else:
                            TDP = RcapHc/K1P
                        CCTest_data_proposed_v3[sim, pi_th_idx, idx_alpha, idx, 1] = TDP

                        bought_v3[sim, pi_th_idx, idx_alpha, idx] = (K-1) - R
        else:
            if pi_choice_type == "beta":
                pi_boundary = pi_param
                pi_null, pi_alt = simulate_contamination_beta(1, 3, 5, 1, pi_boundary, K0, K)
                randomize_array = np.random.permutation(K-1)
                pi_k = np.hstack((pi_null, pi_alt))[randomize_array]
            elif pi_choice_type == "uniform":
                pi_k = np.random.uniform(low=0, high=1, size=(K-1))
            elif pi_choice_type == "discrete_uniform":
                # pi_k = np.random.randint(low=0, high=20, size=(K-1))/60
                pi_k = np.random.randint(low=0, high=11, size=(K-1))/10
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                pi_true[sim, pi_th_idx, :] = pi_k
            print(pi_k)

            data_dict = data_handler.data_splitting(K=K, K0=None, agent_splitting_type=agent_splitting_type, ell0=ell0, n0=n,
                                                    pi_k=pi_k, randomize=randomize, mr0=m, mr1=mr1, T=T, bin_p=bin_p, repeats=repeats, block_size=block_size,
                                                    std=std, distance_cov=distance_cov, std_alt=std_alt, distance_cov_alt=distance_cov_alt, alt_noise_type=alt_noise_type, bin_p_alt=bin_p_alt)

            ### Organize data (part 1) ###
            train_data = data_dict["Agent0_train"][0]
            train_data = np.concatenate((train_data, data_dict["Agent0_calibration"][0]), axis=0)
            train_labels = data_dict["Agent0_train"][1]
            train_labels = np.hstack((train_labels, data_dict["Agent0_calibration"][1]))

            potential_train_data, potential_train_labels, potential_train_null_indicator = import_data_time1(data_dict, data_handler.d, K)

            ### All the data ###
            # allin_train_data, allin_train_labels, allin_train_null_indicator \
            #     = import_data_time2_all(data_dict, K, potential_train_data, potential_train_labels, potential_train_null_indicator)

            ### Conformal scores and p-values ###
            calibration_scores, allin_data_scores = compute_conformal_scores(score_name, score_handler, data_dict,
                                                                             potential_train_data, potential_train_labels,
                                                                             AdaDetect_individual, m, mr1, K-1)
            potential_conformal_pvalues = test_handler.compute_conformal_pvalues(calibration_scores, allin_data_scores)

            # pvals_all[sim, pi_th_idx] = potential_conformal_pvalues.reshape((m, K-1))
            pvals_all = np.hstack((pvals_all, potential_conformal_pvalues))
            indicator_all = np.hstack((indicator_all, potential_train_null_indicator))
            labels_all = np.hstack((labels_all, potential_train_labels))

            ### Run conformal contamination test ###
            test_scores = np.zeros((K-1, m), dtype=np.float64)
            for k in range(1, K):
                # test_scores[k-1] = score_handler.score(data_dict[f"Agent{k}_Time{0}_test"][0], data_dict[f"Agent{k}_Time{0}_test"][1])
                test_scores[k-1] = allin_data_scores[(k-1)*m:k*m]

            pcon = np.zeros((len_pi_th_arr, K-1, 4), dtype=np.float32)
            for pi_th_idx, pi_th in enumerate(pi_th_arr):
                for k in range(1, K):
                    pcon[pi_th_idx, k-1, 0], pcon[pi_th_idx, k-1, 1], pcon[pi_th_idx, k-1, 2], pcon[pi_th_idx, k-1, 3] \
                        = test_handler.all_conformal_contamination_tests(calibration_scores, test_scores[k-1],
                          pi_th=pi_th, n=n, m=m, lambda_=lambda_, i0=i0_arr[pi_th_idx])
                    pcon_total[sim] = pcon

            # ### v1 (theoretically off)
            # rejectBoolCCTest = np.zeros((len(pi_th_arr), K, 4, K-1), dtype=bool)
            # for rho in range(K):
            #     for pi_th_idx, pi_th in enumerate(pi_th_arr):
            #         for idx in range(4):
            #             pvals = pcon[pi_th_idx, :, idx]
            #             sort_ = np.argsort(pvals).astype(np.int16)
            #             pvals_sorted = pvals[sort_]
            #             if rho == K-1:
            #                 pvals_sorted_cutoff = max(0, pvals_sorted[0]-1e-04)
            #                 FDR_est[sim, pi_th_idx, rho, idx] = Storey_FDR(pvals, gamma, pvals_sorted_cutoff)
            #                 rejectBoolCCTest[pi_th_idx, rho, idx, :] = False
            #             elif rho == 0:
            #                 pvals_sorted_cutoff = pvals_sorted[-(rho+1)]
            #                 FDR_est[sim, pi_th_idx, rho, idx] = Storey_FDR(pvals, gamma, pvals_sorted_cutoff)
            #                 rejectBoolCCTest[pi_th_idx, rho, idx, :] = True
            #             else:
            #                 pvals_sorted_cutoff = pvals_sorted[-(rho+1)]
            #                 FDR_est[sim, pi_th_idx, rho, idx] = Storey_FDR(pvals, gamma, pvals_sorted_cutoff)
            #                 rejectBoolCCTest[pi_th_idx, rho, idx, sort_[:-rho]] = True
    
            #             # FDP
            #             rejectBool_ = rejectBoolCCTest[pi_th_idx, rho, idx, :]
            #             R = np.sum(rejectBool_)
            #             RcapH = np.sum(rejectBool_[pi_k<=pi_th])
            #             if R == 0:
            #                 FDP = 0
            #             else:
            #                 FDP = RcapH/R
            #             CCTest_data_proposed[sim, pi_th_idx, rho, idx, 0] = FDP
    
            #             # TDP
            #             K1P = np.sum(pi_k>pi_th)
            #             RcapHc = np.sum(rejectBool_[pi_k>pi_th])
            #             if K1P == 0:
            #                 TDP = 0
            #             else:
            #                 TDP = RcapHc/K1P
            #             CCTest_data_proposed[sim, pi_th_idx, rho, idx, 1] = TDP

            ### v2
            # for idx_reject, reject_threshold in enumerate(rejection_thresholds):
            # # for idx_reject, reject_threshold in enumerate(np.linspace(0, gamma, 11)):
            #     gamma_ = (1+reject_threshold)/2
            #     for pi_th_idx, pi_th in enumerate(pi_th_arr):
            #         for idx in range(4):
            #             pvals = pcon[pi_th_idx, :, idx]
            #             sort_ = np.argsort(pvals).astype(np.int16)
            #             pvals_sorted = pvals[sort_]
            #             rejectBool_ = pvals_sorted <= reject_threshold
            #             # FDR_est_v2[sim, pi_th_idx, idx_reject, idx] = Storey_FDR_CV(pvals, reject_threshold, 100)
            #             FDR_est_v2[sim, pi_th_idx, idx_reject, idx] = Storey_FDR(pvals, gamma_, reject_threshold)
    
            #             # FDP
            #             R = np.sum(rejectBool_)
            #             RcapH = np.sum(rejectBool_[pi_k<=pi_th])
            #             if R == 0:
            #                 FDP = 0
            #             else:
            #                 FDP = RcapH/R
            #             CCTest_data_proposed_v2[sim, pi_th_idx, idx_reject, idx, 0] = FDP
    
            #             # TDP
            #             K1P = np.sum(pi_k>pi_th)
            #             RcapHc = np.sum(rejectBool_[pi_k>pi_th])
            #             if K1P == 0:
            #                 TDP = 0
            #             else:
            #                 TDP = RcapHc/K1P
            #             CCTest_data_proposed_v2[sim, pi_th_idx, idx_reject, idx, 1] = TDP
    
            #             bought_v2[sim, pi_th_idx, idx_reject, idx] = (K-1) - R

            ### Storey's Benjamini-Hochberg
            for idx_alpha, alpha in enumerate(significance_levels):
                for pi_th_idx, pi_th in enumerate(pi_th_arr):
                    for idx in range(4):
                        pvals = pcon[pi_th_idx, :, idx]
                        _, K0_hat = Storeys_correction(gamma, pvals, K-1)
                        rejectBool_ = Benjamini_Hochberg_procedure(pvals, alpha, K0_hat, K-1)

                        # FDP
                        R = np.sum(rejectBool_)
                        RcapH = np.sum(rejectBool_[pi_k<=pi_th])
                        if R == 0:
                            FDP = 0
                        else:
                            FDP = RcapH/R
                        CCTest_data_proposed_v3[sim, pi_th_idx, idx_alpha, idx, 0] = FDP

                        # TDP
                        K1P = np.sum(pi_k>pi_th)
                        RcapHc = np.sum(rejectBool_[pi_k>pi_th])
                        if K1P == 0:
                            TDP = 0
                        else:
                            TDP = RcapHc/K1P
                        CCTest_data_proposed_v3[sim, pi_th_idx, idx_alpha, idx, 1] = TDP

                        bought_v3[sim, pi_th_idx, idx_alpha, idx] = (K-1) - R


    print(tot_time_scores/(sims*len(pi_th_arr)), tot_time_contamination/(sims*len(pi_th_arr)*4))
    np.savez(f"ScoringResults/{savename}.npz", pi_true=pi_true, pvals_all=pvals_all, pi_th_arr=pi_th_arr,
             indicator_all=indicator_all, labels_all=labels_all, pcon_total=pcon_total,
             # FDR_est=FDR_est, CCTest_data_proposed=CCTest_data_proposed,
             # FDR_est_v2=FDR_est_v2, CCTest_data_proposed_v2=CCTest_data_proposed_v2, bought_v2=bought_v2,
             significance_levels=significance_levels, CCTest_data_proposed_v3=CCTest_data_proposed_v3, bought_v3=bought_v3)


if __name__ == "__main__":
    main()
