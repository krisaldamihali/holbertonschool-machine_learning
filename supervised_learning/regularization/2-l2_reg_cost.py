#!/usr/bin/env python3
"""
A script that calculates the cost of a neural
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    A function that calculates the cost of a neural
    """
    regularization_loss = model.losses
    return cost + regularization_loss
