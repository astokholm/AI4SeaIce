#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UNet model."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributor__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.1.2'
__date__ = '2021-14-11'

# -- Third-party modules -- #
import torch
from torch import nn

PARAMETERS = {
    # -- Model parameters -- #
    'n_classes': 12,  # number of total classes in the references
    'kernel_size': (3, 3),  # size of convolutional kernel.
    'stride_rate': (1, 1),  # convolutional striding rate.
    'dilation_rate': (1, 1),
    'padding': (1, 1),  # Number of padding pixels.
    'padding_style': 'zeros',  # Style of applied padding. e.g. zero_pad, replicate.
    'filters': [16, 32, 32, 32]  # Standard 4-level U-Net

    # Used 'filters' parameters:
    #   'filters': [16, 32, 32]  # 3-level
    #   'filters': [16, 32, 32, 32, 32]  # 5-level
    #   'filters': [16, 32, 32, 32, 32, 32]	 # 6-level
    #   'filters': [16, 32, 32, 32, 32, 32, 32]  # 7-level
    #   'filters': [16, 32, 32, 32, 32, 32, 32, 32]	 # 8-level
}


class FeatureMap(nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))


class DoubleConv(nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, parameters, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_n,
                      out_channels=output_n,
                      kernel_size=parameters['kernel_size'],
                      stride=parameters['stride_rate'],
                      padding=parameters['padding'],
                      padding_mode=parameters['padding_style'],
                      bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_n,
                      out_channels=output_n,
                      kernel_size=parameters['kernel_size'],
                      stride=parameters['stride_rate'],
                      padding=parameters['padding'],
                      padding_mode=parameters['padding_style'],
                      bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x


class ContractingBlock(nn.Module):
    """Class to perform downward pass in the U-Net."""

    def __init__(self, parameters, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(parameters=parameters, input_n=input_n, output_n=output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, parameters, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = parameters['padding_style']
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(parameters=parameters, input_n=input_n + output_n, output_n=output_n)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)


def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x


class UNet(nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, parameters):
        super().__init__()

        self.input_block = DoubleConv(parameters=parameters, input_n=parameters['train_variables'].size,
                                      output_n=parameters['filters'][0])

        self.contract_blocks = nn.ModuleList()
        for contract_n in range(1, len(parameters['filters'])):
            self.contract_blocks.append(
                ContractingBlock(parameters=parameters,
                                 input_n=parameters['filters'][contract_n - 1],
                                 output_n=parameters['filters'][contract_n]))  # only used to contract input patch.

        self.bridge = ContractingBlock(
            parameters, input_n=parameters['filters'][-1], output_n=parameters['filters'][-1])

        self.expand_blocks = nn.ModuleList()
        self.expand_blocks.append(ExpandingBlock(parameters=parameters,
                                                 input_n=parameters['filters'][-1],
                                                 output_n=parameters['filters'][-1]))

        for expand_n in range(len(parameters['filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(parameters=parameters,
                                                     input_n=parameters['filters'][expand_n - 1],
                                                     output_n=parameters['filters'][expand_n - 2]))

        self.feature_map = FeatureMap(input_n=parameters['filters'][0], output_n=parameters['n_classes'])

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return self.feature_map(x_expand)


if __name__ == '__main__':
    net = UNet(parameters=PARAMETERS)
