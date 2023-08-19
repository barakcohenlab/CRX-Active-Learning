"""
A model to classify a cis-regulatory sequence as one of four classes: strong enhancer, weak enhancer, inactive, or silencer.
"""

import torch
import numpy as np
import torch.nn as nn 
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class EnhancerModel(nn.Module): 
    """
    A potential enhancer CNN architecture.
    
    Parameters
    ----------
    sequence_length : int
        The length of the sequences used for training and prediction.
    n_targets : int
        The number of targets (classes/labels/activity bins) to predict.
    conv_kernel_size : int
        Size of filters in each convolutional layer.
    pool_kernel_size : int
        Receptive field for max pooling, also the stride length.
    n_filters : int
        Number of convolutional filters in each layer.
    n_fc : int
        Number of neurons in the fully connected layer.
    
    Attributes
    ----------
    conv_net : torch.nn.Sequential
        The convolutional neural network component of the model.
    classifier : torch.nn.Sequential
        The classifier and transformation components of the model.
    """
    
    def __init__(self, sequence_length, n_targets, conv_kernel_size=10, pool_kernel_size=8, n_filters=128, n_fc=32, dropout_c=0.25):
        super(EnhancerModel, self).__init__() 
        self.sequence_length = sequence_length
        self.n_targets = n_targets
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.n_filters = n_filters
        self.n_fc = n_fc
        self.dropout_c = dropout_c
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, self.n_filters, kernel_size=self.conv_kernel_size,
                      padding=self.conv_kernel_size-1),
            nn.BatchNorm1d(self.n_filters),
            nn.LeakyReLU(inplace=True), 
            nn.Dropout(p=dropout_c),
            
            nn.Conv1d(self.n_filters, self.n_filters, kernel_size=self.conv_kernel_size,
                      padding=self.conv_kernel_size-1),
            nn.BatchNorm1d(self.n_filters),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_c),
            
            nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            
            nn.Conv1d(self.n_filters, self.n_filters, kernel_size=self.conv_kernel_size,
                      padding=self.conv_kernel_size-1),
            nn.BatchNorm1d(self.n_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size), 
            nn.Dropout(p=dropout_c)
        )
        
        # Got this from the Sample et al. example https://github.com/FunctionLab/selene/blob/master/tutorials/regression_mpra_example/utr_model.py#L25-L28
        with torch.no_grad():
            clf_input_size = self.conv_net.forward(torch.zeros(1, 4, self.sequence_length)).view(1, -1).shape[1]
            
            
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_size, self.n_fc), 
            nn.BatchNorm1d(self.n_fc),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_c),
            nn.Linear(self.n_fc, self.n_targets), 
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.shape[0], -1) 
        predict = self.classifier(reshape_out)
        return predict


class NLLLoss_(_WeightedLoss):
    # At minimum this should be a subclass of nn.Module, but it should probably be one of nn._WeightedLoss since that's what NLL loss is
    # Normal implementation of NLLLoss: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#NLLLoss
    """
    A custom implementation of NLLLoss that handles multi-class classification with one-hot encoded targets for compatibility with Selene. Suggested by Jian Zhou.

    Parameters
    ----------
    None

    Attributes
    ----------
    nllloss : torch.nn.NLLLoss
        The loss function object.
    """

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super(NLLLoss_, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Use F to call operations here. https://discuss.pytorch.org/t/how-to-choose-between-torch-nn-functional-and-torch-nn-module/2800
        # Decode the one-hot encoded targets
        if target.shape[1] > 1:
            target = target.argmax(1)
        else:
            target = torch.flatten(target).long()
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


def criterion():
    """
    The loss function to be minimized.
    """
    return NLLLoss_()


def get_optimizer(lr):
    """
    The optimizer and the parameters for initialization. Optimizer will be initialized by Selene after model initialization.
    """
    return torch.optim.SGD, {"lr": lr, "momentum": 0.9, "weight_decay": 1e-6}
