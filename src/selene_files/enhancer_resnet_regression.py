"""
A model to predict the activity of a cis-regulatory sequence.
"""

import torch
import numpy as np
import torch.nn as nn 
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class EnhancerResnet(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length

        # FIXME can get from CLI?
        conv_dropout = 0.1
        resid_dropout = 0.25
        fc_neurons = 64
        fc_dropout = 0.25
        # [input_filters, output_filters, filter_size, activation, dilation_layers, pool_size]
        architecture = [
            [4, 128, 10, "exp", 4, 4],   # 164 --> 41
            [128, 128, 6, "relu", 3, 4],    # 41 --> 11
            [128, 64, 3, "relu", 2, 3],   # 11 --> 4
        ]

        layers = []
        for input_filters, output_filters, filter_size, activation, dilation_layers, pool_size in architecture:
            # Conv
            layers.append(nn.Conv1d(
                input_filters, output_filters, kernel_size=filter_size, padding=filter_size-1
            ))
            # BN
            layers.append(nn.BatchNorm1d(output_filters))
            # Activation
            layers.append(get_activation(activation))
            # Dropout
            layers.append(nn.Dropout(p=conv_dropout))
            # Residual dilation
            layers.append(DilationBlock(output_filters, dilation_layers))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(p=resid_dropout))
            # Pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))

        # Flatten + FC
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(fc_neurons))
        # BN
        layers.append(nn.BatchNorm1d(fc_neurons))
        # Activation
        layers.append(nn.ReLU())
        # Dropout
        layers.append(nn.Dropout(p=fc_dropout))

        self.conv_net = nn.Sequential(*layers)
        self.output = nn.Linear(fc_neurons, 1)

    def get_representation(self, x):
        """A forward pass through the CNN portion of the model. This gets the learned representation of the sequence."""
        return self.conv_net(x)

    def forward(self, x):
        x = self.get_representation(x)
        prediction = self.output(x)
        return prediction


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class ResBlock(nn.Module):
    """
    Wrapper to make an arbitrary Sequence a residual block by adding a skip connection.
    https://towardsdatascience.com/building-a-residual-network-with-pytorch-df2f6937053b

    Attributes
    ----------
    block : nn.Sequential
        The arbitrary sequence to wrap in a skip connection.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x.clone() + self.block(x)


class DilationBlock(ResBlock):
    """
    A DilationBlock is a Sequence of convolutions with exponentially increasing dilations. Each convolution is
    followed by batch normalization, activation, and dropout. The final convolution is followed by a batch
    normalization step but not activation or dropout. The entire block is surrounded by a skip connection to create a
    ResBlock. The DilationBlock should be followed by an activation function and dropout after the skip connection.

    Parameters
    ----------
    nfilters : int
        Number of convolution filters to receive as input and use as output.
    nlayers : int
        Number of dilation layers to use.
    rate : int, optional
        Default is 2. Base to use for the dilation factor.
    width : int, optional
        Default is 3. Size of convolution filters.
    activation : str, optional
        Default is "relu". Activation function to use. Other options are "exp" for Exp, "lrelu" for LeakyReLU.
    dropout : float, optional
        Default is 0.1. Dropout rate to use. Must be in the range [0, 1). If 0, no dropout is used.
    """
    def __init__(self, nfilters, nlayers, rate=2, width=3, activation="relu", dropout=0.1):
        layers = []
        # Initial conv and BN
        layers.append(nn.Conv1d(
            nfilters, nfilters, kernel_size=width, padding=width//2, bias=False, dilation=1,
        ))
        layers.append(nn.BatchNorm1d(nfilters))
        for dilation in range(1, nlayers):
            # Activation function
            layers.append(get_activation(activation))
            # Dropout
            if dropout >= 1:
                raise ValueError("Dropout is a number greater than 1.")
            elif dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            # Conv
            layers.append(nn.Conv1d(
                nfilters, nfilters, kernel_size=width, padding=(width // 2 + 1) ** dilation, bias=False,
                dilation=rate ** dilation
            ))
            # BN
            layers.append(nn.BatchNorm1d(nfilters))
        super().__init__(nn.Sequential(*layers))


def get_activation(function):
    """Given a string name of an activation function, return an object for the corresponding Module."""
    if function == "relu":
        return nn.ReLU()
    elif function == "exp":
        return Exp()
    elif function == "lrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError("Did not recognize activation function name.")


def criterion():
    """
    The loss function to be minimized.
    """
    return nn.MSELoss()


def get_optimizer(lr):
    """
    The optimizer and the parameters for initialization. Optimizer will be initialized by Selene after model initialization.
    """
    return torch.optim.Adam, {"lr": lr, "weight_decay": 1e-6}
