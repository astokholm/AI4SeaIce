#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Loss functions."""

# -- File info -- #
__author__ = ['Andrzej S. Kucik', 'Andreas R. Stokholm']
__copyright__ = ['European Space Agency', 'Technical University of Denmark']
__contact__ = ['andrzej.kucik@esa.int', 'stokholm@space.dtu.dk']
__version__ = '0.0.1'
__date__ = '2022-03-31'

# -- Third-party modules -- #
import torch


def labels_to_one_hot(options: dict, labels):
    """
    Convert labels to one-hot-encoded format.
    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    labels :
        ndTensor, true examples with dimensions (batch, height, width).

    Returns
    -------
    labels
        One-hot-encoded with dimensions (batch, n_classes, height, width).
    """
    # (batch, height, # width, n_classes)
    labels = torch.nn.functional.one_hot(labels,
                                         num_classes=options['n_classes'][options['chart']])

    # (batch, n_classes, height, width)
    labels = labels.permute(0, 3, 1, 2).type(torch.float)

    return labels


def bce(options: dict, output, target, bins_w):
    """
    Calculate (weighted) Binary Cross-entropy loss between output and target.

    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    output :
        ndTensor with model output. [batch, width, height]
    target :
        ndTensor with class labels. [batch, width, height]
    bins_w :
        Class weight bins.

    Returns
    -------
    bce :
        The class weighted or averaged bce loss.
    """
    target = target.squeeze()
    output = torch.sigmoid(output.squeeze())

    # Get number of true pixels
    true_pixels = torch.sum(target != options['class_fill_values'][options['chart']])

    # Convert target to [0, 1]
    target_small = torch.clamp(target.type(torch.float) / (options['n_classes'][options['chart']] - 1), max=1.)

    # BCE loss
    bce = torch.sum(bins_w[target] * torch.nn.functional.binary_cross_entropy(
        input=output, target=target_small, reduction='none')) / true_pixels

    return bce


def emd_2(options: dict, output, target, bins_w):
    """
    Calculate the squared Earth Mover's Distance.

    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    output :
        ndTensor with model output. [batch, channel/class, width, height]
    target :
        ndTensor with class labels. [batch, channel/class, width, height]
    bins_w :
        Class weight bins.

    Returns
    -------
    emd2 :
        The class weighted or averaged squared Earth Mover's Distance.
    """
    target = labels_to_one_hot(options, target)
    output = torch.exp(torch.nn.functional.log_softmax(input=output, dim=1))

    # Get number of true pixels
    true_pixels = torch.sum(target != options['class_fill_values'][options['chart']])

    emd2 = torch.sum(bins_w[target.argmax(dim=1)] * torch.sum(
        torch.square(torch.cumsum(target, axis=1) - torch.cumsum(output, axis=1)), axis=1)) / true_pixels

    return emd2


def mse(options: dict, output, target, bins_w):
    """
    Calculate Mean Squared Error (MSE) loss between output and target, weighted with respect to bins_w.
    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    output :
        ndTensor with model output. [batch, width, height]
    target :
        ndTensor with class labels. [batch, width, height]
    bins_w :
        Class weight bins.

    Returns
    -------
    mse :
        The class weighted or averaged MSE loss.
    """
    target = target.squeeze()
    output = output.squeeze()

    # Get number of true pixels
    true_pixels = torch.sum(target != options['class_fill_values'][options['chart']])

    # MSE_loss(i, j) = w_i / sum_w * | output_j - target_j | ^ 2, i = class, j = pixel.
    mse = torch.sum(bins_w[target] * torch.nn.functional.mse_loss(
        input=output, target=target.type(torch.float), reduction='none')) / true_pixels

    return mse

