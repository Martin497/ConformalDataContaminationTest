#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:24:14 2025

This module contains basic functionality for computing conformal scores.
"""


import numpy as np
# import matplotlib.pyplot as plt

import sys, os
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

try:
    from sklearn.svm import OneClassSVM, SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    print("ModuleNotFoundError: No module named 'sklearn'")

try:
    from autoencoder import Autoencoder
    from tensorflow.keras import losses
    from tensorflow import device
except ModuleNotFoundError:
    print("ModuleNotFoundError: No module named 'tensorflow'")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # import torchvision.transforms as transforms
    # from torch.utils.data import Dataset, ConcatDataset
    from torch.utils.data import DataLoader
    from ResNet18 import ResNet_18, ResNet_9, ResNet_4, CNN, train_model, CustomTensorDataset
except ModuleNotFoundError:
    print("ModuleNotFoundError: No module named 'torch'")


class ConformalScore(object):
    """
    Wrapper class for computing conformal scores.

    The class contains functionality for a variety of different data-driven
    conformal scores, some of which are adaptive (AdaDetect).
    Some scores are label-specific, some are purely based on features,
    and some are based jointly on features and labels.
    """
    def __init__(self, type_, **kwargs):
        """
        Initialize the conformal scores.

        Inputs:
        -------
            type_ : str
                Options are 
                    "OCSVM", "labelOCSVM", "OCSVMXY",
                    "LocalOutlierFactor", "labelLocalOutlierFactor", "LocalOutlierFactorXY",
                    "IsolationForest", "labelIsolationForest", "IsolationForestXY",
                    "AdaDetect", "labelAdaDetect", "AdaDetectXY",
                    "Autoencoder", "labelAutoencoder", "AutoencoderXY",
                    "SVC"]
        """
        super(ConformalScore, self).__init__()
        type_options = ["OCSVM", "labelOCSVM", "OCSVMXY",
                        "LocalOutlierFactor", "labelLocalOutlierFactor", "LocalOutlierFactorXY",
                        "IsolationForest", "labelIsolationForest", "IsolationForestXY",
                        "AdaDetect", "labelAdaDetect", "AdaDetectXY",
                        "Autoencoder", "labelAutoencoder", "AutoencoderXY",
                        "SVC"]
        assert type_ in type_options, "The chosen type of conformal score is not supported."
        self.device = "/CPU:0"
        self.type_ = type_
        self.labels = kwargs.get("labels", [i for i in range(10)])
        self.num_labels = len(self.labels)
        self.use_PCA = kwargs.get("use_PCA", False)
        if self.use_PCA is True:
            self.PCAmodel = PCA(kwargs["n_components"])
            kwargs.pop("n_components")

        if (type_ == "OCSVM") or (type_ == "OCSVMXY"):
            self.CSmodel = OneClassSVM()
        elif type_ == "labelOCSVM":
            self.CSmodel = [OneClassSVM() for _ in range(self.num_labels)]
        elif type_ == "SVC":
            self.CSmodel = SVC(probability=True)
        elif type_ == "labelAdaDetect":
            classifier = kwargs["classifier"]
            self.classifier = classifier
            if classifier == "LogisticRegression":
                self.AdaDetect_model = [LogisticRegression(class_weight="balanced") for _ in range(self.num_labels)]
            elif classifier == "SVC":
                self.AdaDetect_model = [SVC(probability=True, class_weight="balanced") for _ in range(self.num_labels)]
            elif classifier == "MLPClassifier":
                self.AdaDetect_model = [MLPClassifier(max_iter=1000, tol=1e-03) for _ in range(self.num_labels)]
            elif classifier == "GradientBoostingClassifier":
                self.AdaDetect_model = [GradientBoostingClassifier() for _ in range(self.num_labels)]
            # self.AdaDetect_classifier = classifier
        elif (type_ == "AdaDetect") or (type_ == "AdaDetectXY"):
            classifier = kwargs["classifier"]
            self.classifier = classifier
            if classifier == "LogisticRegression":
                self.AdaDetect_model = LogisticRegression(class_weight="balanced")
            elif classifier == "SVC":
                self.AdaDetect_model = SVC(probability=True, class_weight="balanced")
            elif classifier == "MLPClassifier":
                self.AdaDetect_model = MLPClassifier(max_iter=1000, tol=1e-03)
                # self.AdaDetect_model.set_fit_request(sample_weight=True)
            elif classifier == "GradientBoostingClassifier":
                self.AdaDetect_model = GradientBoostingClassifier()
                # self.AdaDetect_model.set_fit_request(sample_weight=True)
            elif classifier == "CNN":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.AdaDetect_model = CNN(3, 2)
                self.AdaDetect_model.to(self.device)
                next(self.AdaDetect_model.parameters()).is_cuda
            elif classifier == "ResNet18":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.AdaDetect_model = ResNet_18(3, 2)
                self.AdaDetect_model.to(self.device)
                next(self.AdaDetect_model.parameters()).is_cuda
            elif classifier == "ResNet9":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.AdaDetect_model = ResNet_9(3, 2)
                self.AdaDetect_model.to(self.device)
                next(self.AdaDetect_model.parameters()).is_cuda
            elif classifier == "ResNet4":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.AdaDetect_model = ResNet_4(3, 2)
                self.AdaDetect_model.to(self.device)
                next(self.AdaDetect_model.parameters()).is_cuda
            # self.AdaDetect_classifier = classifier
        elif (type_ == "LocalOutlierFactor") or (type_ == "LocalOutlierFactorXY"):
            self.CSmodel = LocalOutlierFactor(novelty=True)
        elif (type_ == "IsolationForest") or (type_ == "IsolationForestXY"):
            self.CSmodel = IsolationForest()
        elif type_ == "labelLocalOutlierFactor":
            self.CSmodel = [LocalOutlierFactor(novelty=True) for _ in range(self.num_labels)]
        elif type_ == "labelIsolationForest":
            self.CSmodel = [IsolationForest() for _ in range(self.num_labels)]
        elif (type_ == "Autoencoder") or (type_ == "AutoencoderXY"):
            with device(self.device):
                self.CSmodel = Autoencoder(latent_dim=kwargs["latent_dim"], shape=[kwargs["shape"]])
                self.CSmodel.compile(optimizer='adam', loss=losses.MeanSquaredError())
        elif type_ == "labelAutoencoder":
            with device(self.device):
                self.CSmodel = [Autoencoder(latent_dim=kwargs["latent_dim"], shape=[kwargs["shape"]]) for _ in range(self.num_labels)]
                for mod in self.CSmodel:
                    mod.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def one_hot_encoding(self, labels):
        """
        Inputs:
        -------
            labels : ndarray, size=(n,)

        Output:
        -------
            one_hot_labels : ndarray, size=(n, d')
        """
        enc = OneHotEncoder()
        enc.fit(np.expand_dims(labels, axis=1))
        one_hot_labels = enc.transform(np.expand_dims(labels, axis=1)).toarray()
        return one_hot_labels

    def fit(self, X_train, Y_train=None):
        """
        Inputs:
        -------
            X_train : ndarray, size=(ell, d)
                The data used for learning the conformal score.
            Y_train : ndarray, size=(ell,)
                The labels.
        """
        if self.use_PCA is True:
            self.PCAmodel.fit(X_train)
            X_train = self.PCAmodel.transform(X_train)

        if (self.type_ == "OCSVM") or (self.type_ == "LocalOutlierFactor") or (self.type_ == "IsolationForest"):
            self.CSmodel.fit(X_train)
        elif (self.type_ == "OCSVMXY") or (self.type_ == "LocalOutlierFactorXY") or (self.type_ == "IsolationForestXY"):
            Y_train_ohe = self.one_hot_encoding(Y_train)
            XY_train = np.concatenate((X_train, Y_train_ohe), axis=1)
            self.CSmodel.fit(XY_train)
        elif (self.type_ == "labelOCSVM") or (self.type_ == "labelLocalOutlierFactor") or (self.type_ == "labelIsolationForest"):
            for label_idx, label in enumerate(self.labels):
                self.CSmodel[label_idx].fit(X_train[Y_train == label])
        elif self.type_ == "SVC":
            self.CSmodel.fit(X_train, Y_train)
        elif self.type_ == "Autoencoder":
            with device(self.device):
                self.CSmodel.fit(X_train, X_train, epochs=100, shuffle=True, verbose=0)                
        elif self.type_ == "AutoencoderXY":
            Y_train_ohe = self.one_hot_encoding(Y_train)
            XY_train = np.concatenate((X_train, Y_train_ohe), axis=1)
            with device(self.device):
                self.CSmodel.fit(XY_train, XY_train, epochs=100, shuffle=True, verbose=0)                
        elif self.type_ == "labelAutoencoder":
            with device(self.device):
                for label_idx, label in enumerate(self.labels):
                    self.CSmodel[label_idx].fit(X_train[Y_train == label], X_train[Y_train == label], epochs=100, shuffle=True, verbose=0)

    def score(self, X_test, Y_test=None):
        """
        Inputs:
        -------
            X_test : ndarray, size=(n, d)
                Data on which we evaluate the conformal score.
            Y_test : ndarray, size=(n,)
                The labels.
        """
        if self.use_PCA is True:
            X_test = self.PCAmodel.transform(X_test)

        if (self.type_ == "OCSVM") or (self.type_ == "LocalOutlierFactor") or (self.type_ == "IsolationForest"):
            scores = self.CSmodel.score_samples(X_test)
        elif (self.type_ == "OCSVMXY") or (self.type_ == "LocalOutlierFactorXY") or (self.type_ == "IsolationForestXY"):
            Y_test_ohe = self.one_hot_encoding(Y_test)
            XY_test = np.concatenate((X_test, Y_test_ohe), axis=1)
            scores = self.CSmodel.score_samples(XY_test)
        elif (self.type_ == "labelOCSVM") or (self.type_ == "labelLocalOutlierFactor") or (self.type_ == "labelIsolationForest"):
            scores = np.zeros(len(X_test), dtype=np.float32)
            for label_idx, label in enumerate(self.labels):
                if np.sum(Y_test == label) > 0:
                    scores[Y_test == label] = self.CSmodel[label_idx].score_samples(X_test[Y_test == label])
        elif self.type_ == "SVC":
            scores = np.zeros(len(X_test), dtype=np.float32)
            probs = self.CSmodel.predict_proba(X_test)
            for label_idx, label in enumerate(self.labels):
                if np.sum(Y_test == label) > 0:
                    scores[Y_test == label] = (probs[Y_test == label])[:, label_idx]
        elif self.type_ == "Autoencoder":
            with device(self.device):
                encoded = self.CSmodel.encoder(X_test).numpy()
                decoded = self.CSmodel.decoder(encoded).numpy()
            scores = -np.linalg.norm(decoded - X_test, axis=1)
        elif self.type_ == "AutoencoderXY":
            Y_test_ohe = self.one_hot_encoding(Y_test)
            XY_test = np.concatenate((X_test, Y_test_ohe), axis=1)
            with device(self.device):
                encoded = self.CSmodel.encoder(XY_test).numpy()
                decoded = self.CSmodel.decoder(encoded).numpy()
            scores = -np.linalg.norm(decoded - XY_test, axis=1)
        elif self.type_ == "labelAutoencoder":
            scores = np.zeros(len(X_test), dtype=np.float32)
            for label_idx, label in enumerate(self.labels):
                if np.sum(Y_test == label) > 0:
                    with device(self.device):
                        encoded = self.CSmodel[label_idx].encoder(X_test[Y_test == label]).numpy()
                        decoded = self.CSmodel[label_idx].decoder(encoded).numpy()
                    scores[Y_test == label] = -np.linalg.norm(decoded - X_test[Y_test == label], axis=1)
        return scores

    def adaptive_score(self, X_train, X_calibration, X_test, Y_train=None, Y_calibration=None, Y_test=None,
                       feature_dim=(32, 32, 3)):
        """
        Inputs:
        -------
            X_train : ndarray, size=(ell, d)
                The data used for learning the conformal score.
            X_calibration : ndarray, size=(n, d)
                The calibration data.
            X_test : ndarray, size=(m, d)
                The test data.
            Y_train : ndarray, size=(ell,)
                The labels.
            Y_calibration : ndarray, size=(n,)
                The labels.
            Y_test : ndarray, size=(m,)
                The labels.
        """
        if self.use_PCA is True:
            self.PCAmodel.fit(X_train)
            X_train = self.PCAmodel.transform(X_train)
            X_calibration = self.PCAmodel.transform(X_calibration)
            X_test = self.PCAmodel.transform(X_test)

        if self.type_ == "labelAdaDetect":
            scores = np.zeros(len(X_calibration)+len(X_test), dtype=np.float32)
            Y_calibration_and_test = np.hstack((Y_calibration, Y_test))
            for label_idx, label in enumerate(self.labels):
                if np.sum(Y_test == label) > 0:
                    X_train_label = X_train[Y_train==label]
                    X_calibration_and_test_label = np.concatenate((X_calibration[Y_calibration==label],
                                                                   X_test[Y_test==label]), axis=0)
                    train_labels = np.ones(X_train_label.shape[0])
                    calibration_and_test_labels = -np.ones(X_calibration_and_test_label.shape[0])
                    features = np.concatenate((X_train_label, X_calibration_and_test_label), axis=0)
                    labels = np.hstack((train_labels, calibration_and_test_labels))
                    # if (self.classifier == "LogisticRegression") or (self.classifier == "SVC"):
                    #     n_samples = len(labels)
                    #     n_classes = 2
                    #     n_positives = len(train_labels)
                    #     n_negatives = len(calibration_and_test_labels)
                    #     sample_weight = np.hstack((np.ones(n_positives)*n_samples/(n_classes*n_positives), np.ones(n_negatives)*n_samples/(n_classes*n_negatives)))
                    #     self.AdaDetect_model[label_idx].fit(features, labels, sample_weight)
                    # else:
                    self.AdaDetect_model[label_idx].fit(features, labels)
                    assert self.AdaDetect_model[label_idx].classes_[1] == 1, ""
                    scores_label = self.AdaDetect_model[label_idx].predict_proba(X_calibration_and_test_label)[:, 1]
                    scores[Y_calibration_and_test==label] = scores_label
        elif self.type_ == "AdaDetect":
            X_calibration_and_test = np.concatenate((X_calibration, X_test), axis=0)
            train_labels = np.ones(X_train.shape[0])
            features = np.concatenate((X_train, X_calibration_and_test), axis=0)
            # if (self.classifier == "LogisticRegression") or (self.classifier == "SVC"):
            #     n_samples = len(labels)
            #     n_classes = 2
            #     n_positives = len(train_labels)
            #     n_negatives = len(calibration_and_test_labels)
            #     sample_weight = np.hstack((np.ones(n_positives)*n_samples/(n_classes*n_positives), np.ones(n_negatives)*n_samples/(n_classes*n_negatives)))
            #     self.AdaDetect_model.fit(features, labels, sample_weight)
            # else:
            if (self.classifier == "CNN") or (self.classifier == "ResNet18") or (self.classifier == "ResNet9") or (self.classifier == "ResNet4"):
                calibration_and_test_labels = np.zeros(X_calibration_and_test.shape[0])
                labels = np.hstack((train_labels, calibration_and_test_labels))

                labels = torch.from_numpy(labels).type(torch.LongTensor)
                features = np.transpose(np.reshape(features.astype(np.float32), (-1, feature_dim[0], feature_dim[1], feature_dim[2])), (0, 3, 1, 2))
                features = torch.from_numpy(features)
                trainset = CustomTensorDataset(features, labels)
                train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

                X_calibration_and_test = np.transpose(np.reshape(X_calibration_and_test.astype(np.float32), (-1, feature_dim[0], feature_dim[1], feature_dim[2])), (0, 3, 1, 2))
                X_calibration_and_test = torch.from_numpy(X_calibration_and_test)
                calibration_and_test_labels = torch.from_numpy(calibration_and_test_labels).type(torch.LongTensor)
                testset = CustomTensorDataset(X_calibration_and_test, calibration_and_test_labels)
                test_loader = DataLoader(testset, batch_size=32, shuffle=False)

                epochs = 10
                criterion = nn.CrossEntropyLoss(weight = torch.Tensor([len(features) / (2*len(X_calibration_and_test)), len(features) / (2*len(X_train))]))
                optimizer = optim.Adam(self.AdaDetect_model.parameters(), lr=0.0001, weight_decay=1e-4)
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
                self.AdaDetect_model = train_model(self.AdaDetect_model, {"train": train_loader}, criterion, optimizer, lr_scheduler, self.device, epochs)
                scores = np.zeros(len(calibration_and_test_labels), dtype=np.float32)
                self.AdaDetect_model.eval()
                scores_idx = 0
                for inputs, _ in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.AdaDetect_model(inputs)
                    outputs = outputs.to("cpu").detach().numpy()
                    # outputs = outputs[:, 1]/outputs[:, 0]
                    # scores[scores_idx:scores_idx+outputs.shape[0]] = outputs
                    outputs = np.exp(outputs)/np.sum(np.exp(outputs), axis=1)[:, None]
                    scores[scores_idx:scores_idx+outputs.shape[0]] = outputs[:, 1]
                    # outputs = np.exp(outputs)/np.sum(np.exp(outputs), axis=1)[:, None]
                    # scores[scores_idx:scores_idx+outputs.shape[0]] = np.max(outputs, axis=1)
                    # outputs = np.abs(outputs[:, 1] - outputs[:, 0])
                    # scores[scores_idx:scores_idx+outputs.shape[0]] = outputs
                    scores_idx += outputs.shape[0]
                # plt.hist(scores[:len(X_calibration)], color="tab:red", alpha=0.5, density=True)
                # plt.hist(scores[len(X_calibration):], color="tab:blue", alpha=0.5, density=True)
                # plt.show()
            else:
                calibration_and_test_labels = -np.ones(X_calibration_and_test.shape[0])
                labels = np.hstack((train_labels, calibration_and_test_labels))
                self.AdaDetect_model.fit(features, labels)
                scores = self.AdaDetect_model.predict_proba(X_calibration_and_test)[:, 1]
                assert self.AdaDetect_model.classes_[1] == 1, ""
        elif self.type_ == "AdaDetectXY":
            Y_train_ohe = self.one_hot_encoding(Y_train)
            Y_calibration_ohe = self.one_hot_encoding(Y_calibration)
            Y_test_ohe = self.one_hot_encoding(Y_test)
            XY_train = np.concatenate((X_train, Y_train_ohe), axis=1)
            XY_calibration = np.concatenate((X_calibration, Y_calibration_ohe), axis=1)
            XY_test = np.concatenate((X_test, Y_test_ohe), axis=1)
            XY_calibration_and_test = np.concatenate((XY_calibration, XY_test), axis=0)
            train_labels = np.ones(X_train.shape[0])
            calibration_and_test_labels = -np.ones(XY_calibration_and_test.shape[0])
            features = np.concatenate((XY_train, XY_calibration_and_test), axis=0)
            labels = np.hstack((train_labels, calibration_and_test_labels))
            # if (self.classifier == "LogisticRegression") or (self.classifier == "SVC"):
            #     n_samples = len(labels)
            #     n_classes = 2
            #     n_positives = len(train_labels)
            #     n_negatives = len(calibration_and_test_labels)
            #     sample_weight = np.hstack((np.ones(n_positives)*n_samples/(n_classes*n_positives), np.ones(n_negatives)*n_samples/(n_classes*n_negatives)))
            #     self.AdaDetect_model.fit(features, labels, sample_weight)
            # else:
            self.AdaDetect_model.fit(features, labels)
            assert self.AdaDetect_model.classes_[1] == 1, ""
            scores = self.AdaDetect_model.predict_proba(XY_calibration_and_test)[:, 1]
        return scores



