#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:01:06 2025

https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch/notebook
"""

import time
import copy
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import pickle
# from sklearn.model_selection import train_test_split

import torch
# import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class CustomTensorDataset(Dataset):

    def __init__(self, data, labels=None, transform=None):      
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):       
        x = self.data[index]

        if self.transform is not None:
            x = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x

    def __len__(self):    
        return self.data.size(0)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # self.softmax = nn.Softmax(dim=0)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x 

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )

class CNN(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(CNN, self).__init__()
        self.in_channels = 32
        self.hidden_channels = 64

        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.conv2 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=5, stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(self.hidden_channels)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.softmax = nn.Softmax(dim=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hidden_channels, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out_x = out.clone()
        out_x = self.relu(out_x) + residual
        # out = self.relu(out)
        # out += residual
        return out_x


class ResNet_9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, image_channels, num_classes):
        super(ResNet_9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

class ResNet_4(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, image_channels, num_classes):
        super(ResNet_4, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(in_features=64, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, num_epochs=50, verbose=True):

    # torch.autograd.set_detect_anomaly(True)
    # since = time.time()
    # val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # for phase in ['train', 'val']: # Each epoch has a training and validation phase
        for phase in ['train']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]: # Iterate over data

                # inputs = transforms.functional.resize(inputs, (112, 112))
                inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # if phase == 'val': # Adjust learning rate based on val loss
            lr_scheduler.step(epoch_loss)

            if verbose is True:
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        # print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model#, val_acc_history

def score_model(model, device, testloader):
    model.eval()
    acc = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.to("cpu").numpy()
        labels = labels.to("cpu").numpy()
        acc += np.sum(labels == predictions)
    acc /= len(testloader)
    return acc

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

if __name__ == "__main__":
    torch.manual_seed(17)
    nn_model = "CNN"

    null_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    noise_type = "brightness"
    cifar10_in = unpickle("CIFAR10/test_batch")
    cifar10_labels = np.array(cifar10_in[b"labels"]).astype(np.int16)   
    cifar10_data = np.transpose(np.reshape(cifar10_in[b"data"].astype(np.float32)/255, (-1, 32, 32, 3), order="F"), axes=(0, 3, 2, 1))
    indicator_arr = np.logical_or.reduce([cifar10_labels == class_ for class_ in null_classes])
    cifar10_data = cifar10_data[indicator_arr]
    cifar10_labels = cifar10_labels[indicator_arr]

    if nn_model == "ResNet_18":
        model = ResNet_18(3, len(null_classes))
    elif nn_model == "ResNet_9":
        model = ResNet_9(3, len(null_classes))
    elif nn_model == "ResNet_4":
        model = ResNet_4(3, len(null_classes))
    elif nn_model == "CNN":
        model = CNN(3, len(null_classes))

    # submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

    labels = torch.from_numpy(cifar10_labels).type(torch.LongTensor)
    data = torch.from_numpy(cifar10_data)

    # train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size = 0.2, random_state = 42)
    train_data = data[:-1000]
    train_labels = labels[:-1000]
    test_data = data[-1000:]
    test_labels = labels[-1000:]

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomAffine(degrees=20, scale=(1.1, 1.1)),
    #     transforms.RandomCrop((32, 32), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
    #     transforms.ToTensor()
    # ])
    # trainset = ConcatDataset([
    #     CustomTensorDataset(train_data, train_labels),
    #     CustomTensorDataset(train_data, train_labels, transform=transform)
    # ])
    trainset = CustomTensorDataset(train_data, train_labels)
    valset = CustomTensorDataset(test_data, test_labels)
    testset = CustomTensorDataset(test_data, test_labels)

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    count_parameters(model)

    model.to(device)
    next(model.parameters()).is_cuda

    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    # model, _ = train_model(model, {"train": train_loader, "val": val_loader}, criterion, optimizer, epochs)
    model = train_model(model, {"train": train_loader}, criterion, optimizer, lr_scheduler, device, epochs)

    model.eval()
    # labels = []
    for inputs, labels in test_loader:
        # inputs = transforms.functional.resize(inputs, (112, 112))
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.to("cpu").numpy()
        labels = labels.to("cpu").numpy()
        # labels.extend(predictions.numpy())

    # submission['Label'] = labels
    # submission.to_csv('submission.csv', index=False)