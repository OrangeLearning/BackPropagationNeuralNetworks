"""
    This file is to provide loss function and ingrate other file
"""
from BPNeuralNetworkBasic import NeuralNetwork


def BPNeuralNetworkTrainProcess(
        train_data,
        shapes,
        optimizer,
        epochs,
        batch_size,
        eta,
        loss
):
    """
    :param train_data:
    :param shapes:
    :param optimizer:
    :param epochs:
    :param batch_size:
    :param eta:
    :param loss:
    :return:
    """

    nn = NeuralNetwork(shapes=shapes)
    nn = optimizer(nn, train_data, epochs, batch_size, eta, loss)
    return nn


def BPNeuralNetworkTestProcess(nn: NeuralNetwork, test_data):
    res ,cnt = nn.evaluate(test_data)

    print("correct rate: ",float(cnt / len(test_data)))
    # for test_x , test_y in test_data:
    print(res)
    return res