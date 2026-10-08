#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:31:10 2025

This module contains basic functionality for importing data, splitting the
data between agents, splitting into training/calibration(/test), and adding
artificial noise. This module is also responsible for implementing data contamination.
"""

import numpy as np
from copy import deepcopy


class DataHandler(object):
    """
    Wrapper class for for importing data, splitting the data between agents,
    splitting into training/calibration(/test), adding artificial noise,
    and simulating data contamination.
    """
    def __init__(self, type_, feature_noise="train_test", feature_noise_type="GRF_norm", **kwargs):
        """
        Import the dataset and organize the data in class attributes.
        Add feature noise if relevant.

        Inputs:
        -------
            type_ : str
                Options are "MNIST_partial", "FEMNIST_partial",
                "retina","retinalOCT_partial", "WBC_partial".
            feature_noise : str
                Use option "test" when using feature noise and use option
                "train_test" otherwise. Default is "train_test"
            feature_noise_type : str
                Options are "GRF", "bin_p", and "GRF_norm". Default is "GRF_norm".
        """
        super(DataHandler, self).__init__()
        type_options = ["MNIST", "MNIST_partial", "FEMNIST_partial", "retina", "EyePACS",
                        "EyePACS_wtest", "retinalOCT", "retinalOCT_partial", "WBC_partial"]
        assert type_ in type_options, "The chosen data set is not supported."
        self.type_ = type_
        self.feature_noise = feature_noise
        self.feature_noise_type = feature_noise_type
        if type_ == "MNIST_partial":
            classes = kwargs.get("classes", [1, 4, 7])
            self.classes = classes
            train = np.load("MNIST/MNIST_format.npy").astype(np.float32)
            self.train_labels = np.load("MNIST/MNIST_format_labels.npy").astype(np.int16)
            indicator_arr_train = np.logical_or.reduce([self.train_labels == class_ for class_ in classes])
            train = train[indicator_arr_train]
            self.train_data = train / 255
            self.n_train, self.d = self.train_data.shape
            self.num_classes = len(np.unique(self.train_labels))
            test = np.load("MNIST/test_MNIST_format.npy").astype(np.float32)
            self.test_labels = np.load("MNIST/test_MNIST_format_labels.npy").astype(np.int16)
            indicator_arr_test = np.logical_or.reduce([self.test_labels == class_ for class_ in classes])
            test = test[indicator_arr_test]
            self.test_data = test / 255
            self.n_test, _ = self.test_data.shape
        elif type_ == "FEMNIST_partial":
            self.null_classes = kwargs.get("classes", [10, 11, 12])
            self.alt_classes = kwargs.get("alt_classes", [36, 37, 38])
            self.classes = self.null_classes + self.alt_classes
            train = np.load("FEMNIST/femnist_MNIST_format.npy").astype(np.float32)
            self.train_labels = np.load("FEMNIST/femnist_MNIST_format_labels.npy").astype(np.int16)
            indicator_arr_train = np.logical_or.reduce([self.train_labels == class_ for class_ in self.classes])
            train = train[indicator_arr_train]
            self.train_data = train / 255
            self.n_train, self.d = self.train_data.shape
            self.num_classes = len(np.unique(self.train_labels))
            test = np.load("FEMNIST/test_femnist_MNIST_format.npy").astype(np.float32)
            self.test_labels = np.load("FEMNIST/test_femnist_MNIST_format_labels.npy").astype(np.int16)
            indicator_arr_test = np.logical_or.reduce([self.test_labels == class_ for class_ in self.null_classes])
            test = test[indicator_arr_test]
            self.test_data = test / 255
            self.n_test, _ = self.test_data.shape
            self.num_test_classes = len(np.unique(self.test_labels))
        elif type_ == "retina":
            in_data = np.load("medmnist/retinamnist.npz")
            train_images = in_data["train_images"]
            val_images = in_data["val_images"]
            test_images = in_data["test_images"]
            train_labels = in_data["train_labels"]
            val_labels = in_data["val_labels"]
            test_labels = in_data["test_labels"]

            self.train_data = np.concatenate((train_images, val_images), axis=0).reshape((-1, 28**2*3)).astype(np.float32) / 255
            self.train_labels = np.hstack((train_labels[:, 0], val_labels[:, 0])).astype(np.int16)
            self.test_data = test_images.reshape((-1, 28**2*3)).astype(np.float32) / 255
            self.test_labels = test_labels[:, 0].astype(np.int16)
            self.n_train, self.d = self.train_data.shape
            self.n_test, _ = self.test_data.shape
            self.null_classes = [0, 1, 2, 3, 4]
            self.train_data_alt = np.load("EyePACS/eyepacs_MNIST_format.npy").reshape((-1, 28**2*3)).astype(np.float32) / 255
            self.train_labels_alt = np.load("EyePACS/eyepacs_MNIST_format_labels.npy")
            self.n_train_alt = len(self.train_labels_alt)
        elif type_ == "retinalOCT_partial":
            null_classes = kwargs.get("classes", [2, 3])
            self.null_classes = null_classes
            assert 1 not in null_classes, "DME class not contained in retinalOCT_NEH database!"
            in_data = np.load("medmnist/octmnist.npz")
            train_images = in_data["train_images"]
            val_images = in_data["val_images"]
            test_images = in_data["test_images"]
            train_labels = in_data["train_labels"]
            val_labels = in_data["val_labels"]
            test_labels = in_data["test_labels"]

            train_data = np.concatenate((train_images.reshape((-1, 28**2)).astype(np.float32) / 255,
                                         val_images.reshape((-1, 28**2)).astype(np.float32) / 255), axis=0)
            train_labels = np.concatenate((train_labels[:, 0].astype(np.int16),
                                           val_labels[:, 0].astype(np.int16)), axis=0)
            indicator_arr_train = np.logical_or.reduce([train_labels == class_ for class_ in null_classes])
            self.train_data = train_data[indicator_arr_train]
            self.train_labels = train_labels[indicator_arr_train]
            self.n_train, self.d = self.train_data.shape
            self.num_classes = len(np.unique(self.train_labels))

            test_data = test_images.reshape((-1, 28**2)).astype(np.float32) / 255
            test_labels = test_labels[:, 0].astype(np.int16)
            indicator_arr_test = np.logical_or.reduce([test_labels == class_ for class_ in null_classes])
            self.test_data = test_data[indicator_arr_test]
            self.test_labels = test_labels[indicator_arr_test]
            self.n_test, _ = self.test_data.shape
    
            train_data_alt = np.load("RetinalOCT_NEH/retinalOCT_NEH_MNIST_format.npy").reshape((-1, 28**2)).astype(np.float32) / 255
            train_labels_alt = np.load("RetinalOCT_NEH/retinalOCT_NEH_MNIST_format_labels_refactored.npy").astype(np.int16)
            indicator_arr_train_labels_alt = np.logical_or.reduce([train_labels_alt == class_ for class_ in null_classes])
            self.train_data_alt = train_data_alt[indicator_arr_train_labels_alt]
            self.train_labels_alt = train_labels_alt[indicator_arr_train_labels_alt]
            self.n_train_alt = len(self.train_labels_alt)
        elif type_ == "WBC_partial":
            null_classes = kwargs.get("classes", [0, 1, 4, 5])
            self.null_classes = null_classes
            in_data = np.load("medmnist/bloodmnist.npz")
            train_images = in_data["train_images"]
            val_images = in_data["val_images"]
            test_images = in_data["test_images"]
            train_labels = in_data["train_labels"]
            val_labels = in_data["val_labels"]
            test_labels = in_data["test_labels"]

            train_data = np.concatenate((train_images.reshape((-1, 28**2*3)).astype(np.float32) / 255,
                                         val_images.reshape((-1, 28**2*3)).astype(np.float32) / 255), axis=0)
            train_labels = np.concatenate((train_labels[:, 0].astype(np.int16),
                                           val_labels[:, 0].astype(np.int16)), axis=0)
            indicator_arr_train = np.logical_or.reduce([train_labels == class_ for class_ in null_classes])
            self.train_data = train_data[indicator_arr_train]
            self.train_labels = train_labels[indicator_arr_train]
            self.n_train, self.d = self.train_data.shape
            self.num_classes = len(np.unique(self.train_labels))

            test_data = test_images.reshape((-1, 28**2*3)).astype(np.float32) / 255
            test_labels = test_labels[:, 0].astype(np.int16)
            indicator_arr_test = np.logical_or.reduce([test_labels == class_ for class_ in null_classes])
            self.test_data = test_data[indicator_arr_test]
            self.test_labels = test_labels[indicator_arr_test]
            self.n_test, _ = self.test_data.shape
    
            train_data_alt = np.load("WBC/WBC_MNIST_format.npy").reshape((-1, 28**2*3)).astype(np.float32) / 255
            train_labels_alt = np.load("WBC/WBC_MNIST_format_labels.npy").astype(np.int16)
            indicator_arr_train_labels_alt = np.logical_or.reduce([train_labels_alt == class_ for class_ in null_classes])
            self.train_data_alt = train_data_alt[indicator_arr_train_labels_alt]
            self.train_labels_alt = train_labels_alt[indicator_arr_train_labels_alt]
            self.n_train_alt = len(self.train_labels_alt)

        if (type_ != "retina") and (type_ != "EyePACS") and (type_ != "EyePACS_wtest") and (type_ != "WBC_partial"):
            posx = np.arange(28)
            posy = np.arange(28)
            C1, C2 = np.meshgrid(posx, posy)
            pos_flat = np.stack((C1.flatten(), C2.flatten()))
            self.distances = np.linalg.norm(pos_flat[:, :, None] - pos_flat[:, None, :], axis=0)

            if feature_noise == "train_test":
                self.train_data, self.test_data = self.all_feature_noise(feature_noise_type, **kwargs)
            elif feature_noise == "test":
                _, self.test_data = self.all_feature_noise(feature_noise_type, **kwargs)
        else:
            print("RGB data does not currently support feature noise")

    def data_splitting(self, K, K0, agent_splitting_type="class", **kwargs):
        """
        Split the data between the agents and split the data for Agent0
        into training and calibration data.

        Inputs:
        -------
            K : int
                The number of agents.
            K0 : int
                The number of other agents following the null (pi <= pi_th)
            agent_splitting_type : str
                label_noise: add label noise to part of the data. Assign ell0+n0
                             data without label noise to Agent0, assign m0 sampled
                             as binomial(m, 1-pi) to teach of the remaining K-1
                             agents (without label noise), and assign the
                             remaining m-m0 data with label noise.
                feature_noise: add different types of feature noise to inliers
                               and outliers.
                femnist: Use upper case letters as inliers and lower case 
                         letters as outliers.

            kwargs : dict
                ell0 : int
                    Number of training data points. Default is 60.
                n0 : int
                    Number of calibration data points. Default is 40.
                pi_k : ndarray, size=(K,)
                    Array of contamination factors.
                randomize : bool
                    If True randomly shuffle the data. The default is True.
                mr0 : int
                    The number of data points per agent in the first round. The default is 10.
                mr1 : int
                    The number of data points per agent in the second round. The default is 10.
                T : int
                    Number of rounds. The default is 2.
                std : float
                    The distance parameter for "GRF" and "GRF_norm" for the
                    null. The default is 0.2.
                distance_cov : float
                    The distance parameter for "GRF" and "GRF_norm" for the
                    null. The default is 0.1.
                std_alt : float
                    The standard deviation for "GRF" and "GRF_norm" for the
                    alternative/contaminated data. The default is 0.3.
                distance_cov_alt : float
                    The distance parameter for "GRF" and "GRF_norm" for the
                    alternative/contaminated data. The default is 0.1.
                bin_p_alt : float
                    Parameter of "bin_p" noise. The default is 0.7.
                alt_noise_type : str
                    The default is "GRF_norm".
        Output:
        -------
            local_data : dict
                Full data dictionary. The local data is in keys "Agent0_train"
                and "Agent0_calibration". The data of the other data agents is
                in keys "Agent{k}_Time{t}_test" for the k-th data agent and
                t-th round, k=0,dots,K-1, t=0,dots,T-1.
        """
        ell0 = int(kwargs.get("ell0", 100))
        n0 = int(kwargs.get("n0", 500))
        assert ell0 > 5, "Must contain training data."
        assert n0 > 5, "Must contain calibration data."

        ### Keyword arguments ###
        pi_k = kwargs.get("pi_k", np.hstack((0.5  * np.ones(K//2), 0.6 * np.ones(K-1-K//2))))
        randomize = bool(kwargs.get("randomize", True))
        mr0 = int(kwargs.get("mr0", 10))
        mr1 = int(kwargs.get("mr1", 10))
        T = int(kwargs.get("T", 2))
        assert mr0 > 0, "The number of test data points."
        assert mr1 > 0, "The number of test data points."
        assert T > 0, "Number of time epochs most be positive."
        change_pi_k = False

        ### Make sure enough data is available ###
        assert T == 2, ""
        if (agent_splitting_type == "retina") or (agent_splitting_type == "EyePACS") \
        or (agent_splitting_type == "EyePACS_wtest") or (agent_splitting_type == "retinalOCT") \
        or (agent_splitting_type == "WBC") or (agent_splitting_type == "cifar10") \
        or (agent_splitting_type == "imagenet"):
            total_null_data = 1e10
            total_alt_data = 1e10
            tries = 0
            while (self.n_train < total_null_data) or (self.n_train_alt < total_alt_data):
                m0_arr = np.zeros((T, K-1), dtype=np.int32)
                m1_arr = np.zeros((T, K-1), dtype=np.int32)
                for t in range(T):
                    for k in range(1, K):
                        if t == 0:
                            m = mr0
                        elif t == 1:
                            m = mr1
                        m0_arr[t, k-1] = np.random.binomial(m, p=1-pi_k[k-1])
                        m1_arr[t, k-1] = m - m0_arr[t, k-1]
                total_null_data = ell0+n0+np.sum(m0_arr)
                total_alt_data = np.sum(m1_arr)
                tries += 1
                if tries > 1:
                    print("Re-sampling data-splitting", self.n_train, total_null_data, self.n_train_alt, total_alt_data)
                if tries > 5:
                    change_pi_k = True
                    pi_k = np.random.uniform(low=0, high=1, size=(K-1))
                    tries = 0
                    print("Re-sample contamination factors:")
                    print(pi_k)
                if tries > 10:
                    print(self.n_train, total_null_data, self.n_train_alt, total_alt_data)
                    assert False, "Not enough data."
            print(self.n_train, total_null_data, self.n_train_alt, total_alt_data)
            assert self.n_train >= total_null_data, ""
            assert self.n_train_alt >= total_alt_data, ""
        elif (agent_splitting_type == "label_noise") or (agent_splitting_type == "feature_noise"):
            assert ell0+n0+(mr0+mr1)*(K-1) <= self.n_train, "Only so much data is available."
            m0_arr = np.zeros((T, K-1), dtype=np.int32)
            m1_arr = np.zeros((T, K-1), dtype=np.int32)
            for t in range(T):
                for k in range(1, K):
                    if t == 0:
                        m = mr0
                    elif t == 1:
                        m = mr1
                    m0_arr[t, k-1] = np.random.binomial(m, p=1-pi_k[k-1])
                    m1_arr[t, k-1] = m - m0_arr[t, k-1]
        elif agent_splitting_type == "femnist":
            total_null_data = 1e10
            total_alt_data = 1e10
            tries = 0
            while (self.n_train/3 < total_null_data) or (self.n_train/3 < total_alt_data):
                m0_arr = np.zeros((T, K-1), dtype=np.int32)
                m1_arr = np.zeros((T, K-1), dtype=np.int32)
                for t in range(T):
                    for k in range(1, K):
                        if t == 0:
                            m = mr0
                        elif t == 1:
                            m = mr1
                        m0_arr[t, k-1] = np.random.binomial(m, p=1-pi_k[k-1])
                        m1_arr[t, k-1] = m - m0_arr[t, k-1]
                total_null_data = ell0+n0+np.sum(m0_arr)
                total_alt_data = np.sum(m1_arr)
                tries += 1
                if tries > 10:
                    assert False, ""

        if agent_splitting_type == "label_noise":
            assert self.feature_noise == "train_test", ""
            ### Randomization ###
            if randomize is True:
                all_represented = False
                while all_represented is False:
                    num_data = len(self.train_labels)
                    randomize_array = np.random.permutation(num_data)
                    train_data = self.train_data.take(randomize_array, axis=0) # SLOW (but fastest solution I could find even beating a numba implementation)
                    train_labels = self.train_labels.take(randomize_array)
                    ### Agent0 data ###
                    null_data_train = train_data[:ell0]
                    null_labels_train = train_labels[:ell0]
                    null_data_calibration = train_data[ell0:ell0+n0]
                    null_labels_calibration = train_labels[ell0:ell0+n0]
                    local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                    local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})
                    if np.any([len(null_labels_train[null_labels_train == class_]) == 0 for class_ in self.classes]) \
                        or np.any([len(null_labels_calibration[null_labels_calibration == class_]) == 0 for class_ in self.classes]):
                        all_represented = False
                        print("Re-drawing null data sample.")
                    else:
                        all_represented = True
            elif randomize is False:
                train_data, train_labels = self.train_data, self.train_labels
                ### Agent0 data ###
                null_data_train = train_data[:ell0]
                null_labels_train = train_labels[:ell0]
                null_data_calibration = train_data[ell0:ell0+n0]
                null_labels_calibration = train_labels[ell0:ell0+n0]
                local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

            ### Other agents data ###
            data_counter = ell0+n0
            for t in range(T):
                for k in range(1, K):
                    m0 = m0_arr[t, k-1]
                    m1 = m1_arr[t, k-1]
                    m = m0+m1

                    test_data = train_data[data_counter:data_counter+m]
                    test_labels = np.hstack((train_labels[data_counter:data_counter+m0],
                                             np.random.choice(self.classes, size=m1)))
                    equal = np.sum(test_labels[m0:] == train_labels[data_counter+m0:data_counter+m])
                    while equal > 0:
                        test_labels[m0:] = np.where(test_labels[m0:] == train_labels[data_counter+m0:data_counter+m],
                                                    np.random.choice(self.classes, size=m1), test_labels[m0:])
                        equal = np.sum(test_labels[m0:] == train_labels[data_counter+m0:data_counter+m])

                    test_null_indicator = np.hstack((np.zeros(m0, dtype=bool), np.ones(m1, dtype=bool)))
                    local_data.update({f"Agent{k}_Time{t}_test": (test_data, test_labels, test_null_indicator)})
                    data_counter += m
        if agent_splitting_type == "feature_noise":
            assert self.feature_noise == "test", ""
            ### Keyword arguments ###
            std_null = kwargs.get("std", 0.2)
            distance_cov_null = kwargs.get("distance_cov", 0.1)
            std_alt = kwargs.get("std_alt", 0.3)
            distance_cov_alt = kwargs.get("distance_cov_alt", 0.1)
            bin_p_alt = kwargs.get("bin_p_alt", 0.7)
            alt_noise_type = kwargs.get("alt_noise_type", "GRF")

            ### Randomization ###
            if randomize is True:
                all_represented = False
                while all_represented is False:
                    num_data = len(self.train_labels)
                    randomize_array = np.random.permutation(num_data)
                    train_data = self.train_data.take(randomize_array, axis=0) # SLOW (but fastest solution I could find even beating a numba implementation)
                    train_labels = self.train_labels.take(randomize_array)
                    ### Agent0 data ###
                    null_data_train = self.feature_noise_fun(train_data[:ell0], feature_noise_type=self.feature_noise_type, std=std_null, distance_cov=distance_cov_null)
                    null_labels_train = train_labels[:ell0]
                    null_data_calibration = self.feature_noise_fun(train_data[ell0:ell0+n0], feature_noise_type=self.feature_noise_type, std=std_null, distance_cov=distance_cov_null)
                    null_labels_calibration = train_labels[ell0:ell0+n0]
                    local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                    local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})
                    if np.any([len(null_labels_train[null_labels_train == class_]) == 0 for class_ in self.classes]) \
                        or np.any([len(null_labels_calibration[null_labels_calibration == class_]) == 0 for class_ in self.classes]):
                        all_represented = False
                        print("Re-drawing null data sample.")
                    else:
                        all_represented = True
            elif randomize is False:
                train_data, train_labels = self.train_data, self.train_labels
                ### Agent0 data ###
                null_data_train = self.feature_noise_fun(train_data[:ell0], feature_noise_type="GRF", std=std_null, distance_cov=distance_cov_null)
                null_labels_train = train_labels[:ell0]
                null_data_calibration = self.feature_noise_fun(train_data[ell0:ell0+n0], feature_noise_type="GRF", std=std_null, distance_cov=distance_cov_null)
                null_labels_calibration = train_labels[ell0:ell0+n0]
                local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

            ### Other agents data ###
            data_counter = ell0+n0
            for t in range(T):
                for k in range(1, K):
                    m0 = m0_arr[t, k-1]
                    m1 = m1_arr[t, k-1]
                    m = m0+m1

                    test_data = np.zeros((m, 28**2), dtype=np.float32)
                    test_data[:m0] = self.feature_noise_fun(train_data[data_counter:data_counter+m0], feature_noise_type=self.feature_noise_type, std=std_null, distance_cov=distance_cov_null)
                    if alt_noise_type == "GRF" or alt_noise_type == "GRF_norm":
                        test_data[m0:] = self.feature_noise_fun(train_data[data_counter+m0:data_counter+m], feature_noise_type=alt_noise_type, std=std_alt, distance_cov=distance_cov_alt)
                    elif alt_noise_type == "bin_p":
                        test_data[m0:] = self.feature_noise_fun(train_data[data_counter+m0:data_counter+m], feature_noise_type=alt_noise_type, bin_p=bin_p_alt)
                    test_labels = train_labels[data_counter:data_counter+m]

                    test_null_indicator = np.hstack((np.zeros(m0, dtype=bool), np.ones(m1, dtype=bool)))
                    local_data.update({f"Agent{k}_Time{t}_test": (test_data, test_labels, test_null_indicator)})
                    data_counter += m
        if agent_splitting_type == "femnist":
            assert self.feature_noise == "train_test", ""

            ### Randomization ###
            if randomize is True:
                all_represented = False
                while all_represented is False:
                    num_data = len(self.train_labels)
                    randomize_array = np.random.permutation(num_data)
                    train_data = self.train_data.take(randomize_array, axis=0) # SLOW (but fastest solution I could find even beating a numba implementation)
                    train_labels = self.train_labels.take(randomize_array)

                    train_data_fromnull = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
                    train_labels_fromnull = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
                    train_data_fromalt = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]
                    train_labels_fromalt = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]

                    ### Agent0 data ###
                    null_data_train = train_data_fromnull[:ell0]
                    null_labels_train = train_labels_fromnull[:ell0]
                    null_data_calibration = train_data_fromnull[ell0:ell0+n0]
                    null_labels_calibration = train_labels_fromnull[ell0:ell0+n0]
                    local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                    local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

                    if np.any([len(null_labels_train[null_labels_train == class_]) == 0 for class_ in self.null_classes]) \
                        or np.any([len(null_labels_calibration[null_labels_calibration == class_]) == 0 for class_ in self.null_classes]):
                        all_represented = False
                        print("Re-drawing null data sample.")
                    else:
                        all_represented = True
            elif randomize is False:
                train_data = self.train_data
                train_labels = self.train_labels

                train_data_fromnull = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
                train_labels_fromnull = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
                train_data_fromalt = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]
                train_labels_fromalt = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]

                ### Agent0 data ###
                null_data_train = train_data_fromnull[:ell0]
                null_labels_train = train_labels_fromnull[:ell0]
                null_data_calibration = train_data_fromnull[ell0:ell0+n0]
                null_labels_calibration = train_labels_fromnull[ell0:ell0+n0]
                local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

            ### Other agents data ###
            data_counter_fromnull = ell0+n0
            data_counter_fromalt = 0
            for t in range(T):
                for k in range(1, K):
                    m0 = m0_arr[t, k-1]
                    m1 = m1_arr[t, k-1]
                    m = m0+m1

                    test_data = np.zeros((m, 28**2), dtype=np.float32)
                    test_data[:m0] = train_data_fromnull[data_counter_fromnull:data_counter_fromnull+m0]
                    test_data[m0:] = train_data_fromalt[data_counter_fromalt:data_counter_fromalt+m1]
                    test_labels = np.zeros(m, dtype=np.int16)
                    test_labels[:m0] = train_labels_fromnull[data_counter_fromnull:data_counter_fromnull+m0]
                    test_labels[m0:] = train_labels_fromalt[data_counter_fromalt:data_counter_fromalt+m1]
                    for idx, alt_class in enumerate(self.alt_classes):
                        test_labels[test_labels == alt_class] = self.null_classes[idx]

                    test_null_indicator = np.hstack((np.zeros(m0, dtype=bool), np.ones(m1, dtype=bool)))
                    local_data.update({f"Agent{k}_Time{t}_test": (test_data, test_labels, test_null_indicator)})
                    data_counter_fromnull += m0
                    data_counter_fromalt += m1

        if (agent_splitting_type == "retina") or (agent_splitting_type == "EyePACS") \
        or (agent_splitting_type == "EyePACS_wtest") or (agent_splitting_type == "retinalOCT") \
        or (agent_splitting_type == "WBC") or (agent_splitting_type == "cifar10") \
        or (agent_splitting_type == "imagenet"):
            assert self.feature_noise == "train_test", ""

            ### Randomization ###
            if randomize is True:
                all_represented = False
                while all_represented is False:
                    num_data = len(self.train_labels)
                    randomize_array = np.random.permutation(num_data)
                    train_data_fromnull = self.train_data.take(randomize_array, axis=0) # SLOW (but fastest solution I could find even beating a numba implementation)
                    train_labels_fromnull = self.train_labels.take(randomize_array)

                    num_data_alt = len(self.train_labels_alt)
                    randomize_array_alt = np.random.permutation(num_data_alt)
                    train_data_fromalt = self.train_data_alt.take(randomize_array_alt, axis=0) # SLOW (but fastest solution I could find even beating a numba implementation)
                    train_labels_fromalt = self.train_labels_alt.take(randomize_array_alt)

                    ### Agent0 data ###
                    null_data_train = train_data_fromnull[:ell0]
                    null_labels_train = train_labels_fromnull[:ell0]
                    null_data_calibration = train_data_fromnull[ell0:ell0+n0]
                    null_labels_calibration = train_labels_fromnull[ell0:ell0+n0]
                    local_data = {"Agent0_train": (null_data_train, null_labels_train)}
                    local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

                    if agent_splitting_type == "EyePACS_wtest":
                        all_represented = True
                    else:
                        if np.any([len(null_labels_train[null_labels_train == class_]) == 0 for class_ in self.null_classes]) \
                            or np.any([len(null_labels_calibration[null_labels_calibration == class_]) == 0 for class_ in self.null_classes]):
                            all_represented = False
                            print("Re-drawing null data sample.")
                        else:
                            all_represented = True
            # elif randomize is False:
            #     train_data = self.train_data
            #     train_labels = self.train_labels

            #     train_data_fromnull = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
            #     train_labels_fromnull = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.null_classes])]
            #     train_data_fromalt = train_data[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]
            #     train_labels_fromalt = train_labels[np.logical_or.reduce([train_labels == class_ for class_ in self.alt_classes])]

            #     ### Agent0 data ###
            #     null_data_train = train_data_fromnull[:ell0]
            #     null_labels_train = train_labels_fromnull[:ell0]
            #     null_data_calibration = train_data_fromnull[ell0:ell0+n0]
            #     null_labels_calibration = train_labels_fromnull[ell0:ell0+n0]
            #     local_data = {"Agent0_train": (null_data_train, null_labels_train)}
            #     local_data.update({"Agent0_calibration": (null_data_calibration, null_labels_calibration)})

            ### Other agents data ###
            data_counter_fromnull = ell0+n0
            data_counter_fromalt = 0
            for t in range(T):
                for k in range(1, K):
                    m0 = m0_arr[t, k-1]
                    m1 = m1_arr[t, k-1]
                    m = m0+m1

                    if agent_splitting_type == "retinalOCT":
                        test_data = np.zeros((m, 28**2), dtype=np.float32)
                    elif agent_splitting_type == "cifar10":
                        test_data = np.zeros((m, 32**2*3), dtype=np.float32)
                    elif agent_splitting_type == "imagenet":
                        test_data = np.zeros((m, 64**2*3), dtype=np.float32)
                    else:
                        test_data = np.zeros((m, 28**2*3), dtype=np.float32)
                    test_data[:m0] = train_data_fromnull[data_counter_fromnull:data_counter_fromnull+m0]
                    test_data[m0:] = train_data_fromalt[data_counter_fromalt:data_counter_fromalt+m1]
                    test_labels = np.zeros(m, dtype=np.int16)
                    test_labels[:m0] = train_labels_fromnull[data_counter_fromnull:data_counter_fromnull+m0]
                    test_labels[m0:] = train_labels_fromalt[data_counter_fromalt:data_counter_fromalt+m1]

                    test_null_indicator = np.hstack((np.zeros(m0, dtype=bool), np.ones(m1, dtype=bool)))
                    local_data.update({f"Agent{k}_Time{t}_test": (test_data, test_labels, test_null_indicator)})
                    data_counter_fromnull += m0
                    data_counter_fromalt += m1

        if change_pi_k is True:
            local_data["pi_k"] = pi_k
        self.local_data = deepcopy(local_data)
        return local_data

    def return_test_data(self):
        """
        """
        return self.test_data, self.test_labels

    def feature_noise_fun(self, features, feature_noise_type, **kwargs):
        """
        Add feature noise to individual feature vectors.

        Inputs:
        -------
            features : ndarray, size=(n, 28^2)
                The feature data.
            feature_noise_type : str
                The type of feature noise.

        Output:
        -------
            features_noisy : ndarray, size=(n, 28^2)
                The noisy feature data.
        """
        if (self.type_ == "MNIST") or (self.type_ == "MNIST_partial") \
        or (self.type_ == "FEMNIST") or (self.type_ == "FEMNIST_partial") \
        or (self.type_ == "retinalOCT") or (self.type_ == "retinalOCT_partial"):
            if feature_noise_type == "bin_p":
                bin_p = kwargs.get("bin_p", 0.2)
                n = features.shape[0]
    
                indicator = np.random.binomial(1, bin_p, size=(n*28**2)).astype(bool)
                noisy_val = np.random.uniform(low=0, high=1, size=(n*28**2))
                features_flat = np.copy(features.flatten())
                features_flat[indicator] = noisy_val[indicator]
                features_noisy = features_flat.reshape((n, 28**2))
            elif feature_noise_type == "blocks":
                repeats = kwargs.get("repeats", 2)
                block_size = kwargs.get("block_size", 2)
                n = features.shape[0]

                features_noisy = np.zeros((n, 28, 28), dtype=np.float32)
                for i in range(n):
                    features_noisy[i] = features[i].reshape((28, 28))
                    for j in range(repeats):
                        posx = np.random.randint(low=0, high=28-block_size+1)
                        posy = np.random.randint(low=0, high=28-block_size+1)
                        features_noisy[i, posx:posx+block_size, posy:posy+block_size] \
                            = np.ones((block_size, block_size), dtype=np.float32) * np.random.uniform(low=0, high=1)
                features_noisy = features_noisy.reshape((n, 28**2))
            elif feature_noise_type == "GRF":
                std = kwargs.get("std", 1)
                distance_cov = kwargs.get("distance_cov", 1)
                if std == 0:
                    return features
                n = features.shape[0]
                cov = std**2 * np.exp(-self.distances*distance_cov)

                features_noisy = np.zeros((n, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=n)
                for i in range(n):
                    features_noisy[i] = features[i] + noise[i]
            elif feature_noise_type == "GRF_norm":
                std = kwargs.get("std", 1)
                distance_cov = kwargs.get("distance_cov", 1)
                if std == 0:
                    return features
                n = features.shape[0]
                cov = std**2 * np.exp(-self.distances*distance_cov)

                features_noisy = np.zeros((n, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=n)
                features_noisy = features + noise
                features_noisy = features_noisy - np.min(features_noisy, axis=1)[:, None]
                features_noisy = features_noisy/np.max(features_noisy, axis=1)[:, None]
            return features_noisy

    def all_feature_noise(self, feature_noise_type, **kwargs):
        """
        Add feature noise to entire data.

        Inputs:
        -------
            features_noise_type : str
                The type of feature noise.

        Outputs:
        --------
            train_data_noisy : ndarray, size=(n, 28^2)
                The noisy training data.
            test_data_noisy : ndarray, size=(n, 28^2)
                The noisy test data.
        """
        if (self.type_ == "MNIST") or (self.type_ == "MNIST_partial") \
        or (self.type_ == "FEMNIST") or (self.type_ == "FEMNIST_partial") \
        or (self.type_ == "retinalOCT") or (self.type_ == "retinalOCT_partial"):
            if self.feature_noise_type == "bin_p":
                bin_p = kwargs.get("bin_p", 0.2)

                indicator_train = np.random.binomial(1, bin_p, size=(self.n_train*28**2)).astype(bool)
                noisy_val_train = np.random.uniform(low=0, high=1, size=(self.n_train*28**2))
                train_data_flat = np.copy(self.train_data.flatten())
                train_data_flat[indicator_train] = noisy_val_train[indicator_train]
                train_data_noisy = train_data_flat.reshape((self.n_train, 28**2))

                indicator_test = np.random.binomial(1, bin_p, size=(self.n_test*28**2)).astype(bool)
                noisy_val_test = np.random.uniform(low=0, high=1, size=(self.n_test*28**2))
                test_data_flat = np.copy(self.test_data.flatten())
                test_data_flat[indicator_test] = noisy_val_test[indicator_test]
                test_data_noisy = test_data_flat.reshape((self.n_test, 28**2))
            elif self.feature_noise_type == "blocks":
                repeats = kwargs.get("repeats", 2)
                block_size = kwargs.get("block_size", 2)

                train_data_noisy = np.zeros((self.n_train, 28, 28), dtype=np.float32)
                for i in range(self.n_train):
                    train_data_noisy[i] = self.train_data[i].reshape((28, 28))
                    for j in range(repeats):
                        posx = np.random.randint(low=0, high=28-block_size+1)
                        posy = np.random.randint(low=0, high=28-block_size+1)
                        train_data_noisy[i, posx:posx+block_size, posy:posy+block_size] \
                            = np.ones((block_size, block_size), dtype=np.float32) * np.random.uniform(low=0, high=1)
                train_data_noisy = train_data_noisy.reshape((self.n_train, 28**2))

                test_data_noisy = np.zeros((self.n_test, 28, 28), dtype=np.float32)
                for i in range(self.n_test):
                    test_data_noisy[i] = self.test_data[i].reshape((28, 28))
                    for j in range(repeats):
                        posx = np.random.randint(low=0, high=28-block_size+1)
                        posy = np.random.randint(low=0, high=28-block_size+1)
                        test_data_noisy[i, posx:posx+block_size, posy:posy+block_size] \
                            = np.ones((block_size, block_size), dtype=np.float32) * np.random.uniform(low=0, high=1)
                test_data_noisy = test_data_noisy.reshape((self.n_test, 28**2))
            elif self.feature_noise_type == "GRF":
                std = kwargs.get("std", 1)
                distance_cov = kwargs.get("distance_cov", 1)
                if std == 0:
                    return self.train_data, self.test_data
                cov = std**2 * np.exp(-self.distances*distance_cov)

                train_data_noisy = np.zeros((self.n_train, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=self.n_train)
                for i in range(self.n_train):
                    train_data_noisy[i] = self.train_data[i] + noise[i]

                test_data_noisy = np.zeros((self.n_test, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=self.n_test)
                for i in range(self.n_test):
                    test_data_noisy[i] = self.test_data[i] + noise[i]
            elif self.feature_noise_type == "GRF_norm":
                std = kwargs.get("std", 1)
                distance_cov = kwargs.get("distance_cov", 1)
                if std == 0:
                    return self.train_data, self.test_data
                cov = std**2 * np.exp(-self.distances*distance_cov)

                train_data_noisy = np.zeros((self.n_train, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=self.n_train)
                train_data_noisy = self.train_data + noise
                train_data_noisy = train_data_noisy - np.min(train_data_noisy, axis=1)[:, None]
                train_data_noisy = train_data_noisy/np.max(train_data_noisy, axis=1)[:, None]

                test_data_noisy = np.zeros((self.n_test, 28**2), dtype=np.float32)
                noise = np.random.multivariate_normal(np.zeros(28**2), cov, size=self.n_test)
                test_data_noisy = self.test_data + noise
                test_data_noisy = test_data_noisy - np.min(test_data_noisy, axis=1)[:, None]
                test_data_noisy = test_data_noisy/np.max(test_data_noisy, axis=1)[:, None]
        return train_data_noisy, test_data_noisy
