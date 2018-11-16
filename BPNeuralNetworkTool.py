"""
    This file is to define some aux function

    These function will not change in short time .
"""
import numpy as np
import random
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mini_batch(train_data, mini_batch_size, shuffle=True):
    if shuffle:
        random.seed(time.time())
        random.shuffle(train_data)

    n = len(train_data)
    res = []
    for k in range(0, n, mini_batch_size):
        res.append(train_data[k: k + mini_batch_size])

    return res


def squaredLoss(y0, y):
    return (y - y0) ** 2
