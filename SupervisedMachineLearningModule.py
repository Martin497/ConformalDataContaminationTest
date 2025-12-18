#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:04:49 2025

This module contains basic functionality for fitting and testing supervised
machine learning models for classification.
"""

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    print("ModuleNotFoundError: No module named 'sklearn'")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    from ResNet18 import ResNet_18, ResNet_9, ResNet_4, CNN, train_model, CustomTensorDataset, score_model
except ModuleNotFoundError:
    print("ModuleNotFoundError: No module named 'torch'")


class SupervisedMachineLearning(object):
    """
    Wrapper class for fitting and testing supervised models for classification.
    """
    def __init__(self, type_, **kwargs):
        """
        Inputs:
        -------
            type_ : str
                Options are "LogisticRegression", "KNeighborsClassifier", "SVC", "LinearSVC",
                "MLPClassifier", "GradientBoostingClassifier", "PCA_LogisticRegression", "PCA_SVC".
        """
        super(SupervisedMachineLearning, self).__init__()
        type_options = ["LogisticRegression", "KNeighborsClassifier", "SVC", "LinearSVC",
                        "MLPClassifier", "GradientBoostingClassifier", "PCA_LR",
                        "PCA_SVC", "ResNet18", "ResNet9", "ResNet4", "CNN"]
        assert type_ in type_options, "The chosen supervised machine learning model is not supported."
        self.type_ = type_
        if type_ == "LogisticRegression":
            self.SMLmodel = LogisticRegression(**kwargs)
        elif type_ == "KNeighborsClassifier":
            self.SMLmodel = KNeighborsClassifier(**kwargs)
        elif type_ == "SVC":
            self.SMLmodel = make_pipeline(StandardScaler(), SVC(**kwargs))
        elif type_ == "LinearSVC":
            self.SMLmodel = LinearSVC(**kwargs)
        elif type_ == "MLPClassifier":
            self.SMLmodel = MLPClassifier(**kwargs)
        elif type_ == "GradientBoostingClassifier":
            self.SMLmodel = GradientBoostingClassifier(**kwargs)
        elif type_ == "PCA_LR":
            self.PCAmodel = PCA(kwargs["n_components"])
            kwargs.pop("n_components")
            self.SMLmodel = LogisticRegression(**kwargs)
        elif type_ == "PCA_SVC":
            self.PCAmodel = PCA(kwargs["n_components"])
            kwargs.pop("n_components")
            self.SMLmodel = make_pipeline(StandardScaler(), SVC(**kwargs))
        elif type_ == "ResNet19":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.SMLmodel = ResNet_18(3, kwargs["num_classes"])
            kwargs.pop("num_classes")
            self.SMLmodel.to(self.device)
            next(self.SMLmodel.parameters()).is_cuda
        elif type_ == "ResNet9":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.SMLmodel = ResNet_9(3, kwargs["num_classes"])
            kwargs.pop("num_classes")
            self.SMLmodel.to(self.device)
            next(self.SMLmodel.parameters()).is_cuda
        elif type_ == "ResNet4":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.SMLmodel = ResNet_4(3, kwargs["num_classes"])
            kwargs.pop("num_classes")
            self.SMLmodel.to(self.device)
            next(self.SMLmodel.parameters()).is_cuda
        elif type_ == "CNN":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.SMLmodel = CNN(3, kwargs["num_classes"])
            kwargs.pop("num_classes")
            self.SMLmodel.to(self.device)
            next(self.SMLmodel.parameters()).is_cuda


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

    def zero_sort_encoding(self, labels):
        """
        Inputs:
        -------
            labels : ndarray, size=(n,)

        Output:
        -------
            zero_sort_labels : ndarray, size=(n)
        """
        enc = OneHotEncoder()
        enc.fit(np.expand_dims(labels, axis=1))
        one_hot_labels = enc.transform(np.expand_dims(labels, axis=1)).toarray()
        zero_sort_labels = np.argmax(one_hot_labels, axis=1)
        return zero_sort_labels

    def fit(self, train_features, train_labels, feature_dim=(32, 32, 3)):
        """
        Inputs:
        -------
            train_features : ndarray, size=(n, d)
                The input features used for fitting the supervised model.
            train_labels : ndarray, size=(n,)
                The labels used for fitting the supervised model.
        """
        self.labels = np.unique(train_labels)
        if (self.type_ == "LogisticRegression")  or (self.type_ == "KNeighborsClassifier") \
        or (self.type_ == "SVC") or (self.type_ == "LinearSVC") or (self.type_ == "GradientBoostingClassifier"):
            self.SMLmodel.fit(train_features, train_labels)
        elif self.type_ == "MLPClassifier":
            one_hot_labels = self.one_hot_encoding(train_labels)
            self.SMLmodel.fit(train_features, one_hot_labels)
        elif (self.type_ == "PCA_LR") or (self.type_ == "PCA_SVC"):
            self.PCAmodel.fit(train_features)
            train_PCA = self.PCAmodel.transform(train_features)
            self.SMLmodel.fit(train_PCA, train_labels)
        elif (self.type_ == "CNN") or (self.type_ == "ResNet18") \
        or (self.type_ == "ResNet9") or (self.type_ == "ResNet4"):
            zero_sort_labels = self.zero_sort_encoding(train_labels)
            zero_sort_labels = torch.from_numpy(zero_sort_labels).type(torch.LongTensor)
            train_features = np.transpose(np.reshape(train_features.astype(np.float32),
                                (-1, feature_dim[0], feature_dim[1], feature_dim[2])), (0, 3, 1, 2))
            train_features = torch.from_numpy(train_features)
            trainset = CustomTensorDataset(train_features, zero_sort_labels)
            train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
            epochs = 50
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.SMLmodel.parameters(), lr=0.0001, weight_decay=1e-4)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            self.SMLmodel = train_model(self.SMLmodel, {"train": train_loader}, criterion,
                                        optimizer, lr_scheduler, self.device, epochs, verbose=False)

    def predict(self, test_features):
        """
        Inputs:
        -------
            X_test : ndarray, size=(n, d)
                Data on which we evaluate the conformal score.
        """
        if (self.type_ == "LogisticRegression") or (self.type_ == "KNeighborsClassifier") or (self.type_ == "SVC") \
        or (self.type_ == "LinearSVC") or (self.type_ == "MLPClassifier") or (self.type_ == "GradientBoostingClassifier"):
            predictions = self.SMLmodel.predict(test_features)
        elif (self.type_ == "PCA_LR") or (self.type_ == "PCA_SVC"):
            predictions = self.SMLmodel.predict(self.PCAmodel.transform(test_features))
        return predictions

    def softmax_scores(self, test_features):
        """
        """
        if (self.type_ == "LogisticRegression") or (self.type_ == "KNeighborsClassifier") or (self.type_ == "SVC") \
        or (self.type_ == "LinearSVC") or (self.type_ == "MLPClassifier") or (self.type_ == "GradientBoostingClassifier"):
            probs = self.SMLmodel.predict_proba(test_features)
        elif (self.type_ == "PCA_LR") or (self.type_ == "PCA_SVC"):
            probs = self.SMLmodel.predict_proba(self.PCAmodel.transform(test_features))
        return probs

    def score(self, test_features, test_labels, feature_dim=(32, 32, 3)):
        """
        """
        assert np.all(self.labels == np.unique(test_labels)), "Matching training and test labels!"
        if (self.type_ == "LogisticRegression") or (self.type_ == "KNeighborsClassifier") or (self.type_ == "SVC") \
        or (self.type_ == "LinearSVC") or (self.type_ == "GradientBoostingClassifier"):
            score = self.SMLmodel.score(test_features, test_labels)
        elif self.type_ == "MLPClassifier":
            one_hot_labels = self.one_hot_encoding(test_labels)
            score = self.SMLmodel.score(test_features, one_hot_labels)
        elif (self.type_ == "PCA_LR") or (self.type_ == "PCA_SVC"):
            score = self.SMLmodel.score(self.PCAmodel.transform(test_features), test_labels)
        elif (self.type_ == "CNN") or (self.type_ == "ResNet18") \
        or (self.type_ == "ResNet9") or (self.type_ == "ResNet4"):
            zero_sort_labels = self.zero_sort_encoding(test_labels)
            zero_sort_labels = torch.from_numpy(zero_sort_labels).type(torch.LongTensor)
            test_features = np.transpose(np.reshape(test_features.astype(np.float32),
                               (-1, feature_dim[0], feature_dim[1], feature_dim[2])), (0, 3, 1, 2))
            test_features = torch.from_numpy(test_features)
            testset = CustomTensorDataset(test_features, zero_sort_labels)
            test_loader = DataLoader(testset, batch_size=32, shuffle=False)
            score = score_model(self.SMLmodel, self.device, test_loader)
        return score
