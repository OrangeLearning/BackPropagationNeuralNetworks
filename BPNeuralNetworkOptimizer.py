"""
    This file is to define the Optimizer
"""
from BPNeuralNetworkTool import mini_batch


def SGD(nn, training_data, epochs, batch_size, eta, loss):
    """
        This function is Stochastic Gradient Descent Algorithm

        :param nn: the neural network
        :param training_data:
        :param epochs: iter number
        :param batch_size:
        :param eta: learning rate
        :param loss: loss function
        :return:
    """
    for iter in range(epochs):
        mini_batches = mini_batch(training_data, batch_size, shuffle=True)

        for data in mini_batches:
            nn.updateStep(data, eta, loss)

    return nn
