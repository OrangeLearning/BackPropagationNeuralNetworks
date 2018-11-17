"""
    This file is to integrate all the function
"""
from BPNeuralNetworkData import getIrisData
from BPNeuralNetworkData import divide_train_test
from BPNeuralNetwork import BPNeuralNetworkTrainProcess
from BPNeuralNetworkOptimizer import SGD
from BPNeuralNetworkTool import squaredLoss
from BPNeuralNetwork import BPNeuralNetworkTestProcess


def main():
    data, yLabel, rLabel = getIrisData()
    # yLabel and rLabel is used map string to int

    ttr = 0.8  # train_test_rate
    train_data, test_data = divide_train_test(data, ttr)
    nn = BPNeuralNetworkTrainProcess(data, [4, 5, 1], SGD, 15, 8, 0.1, squaredLoss)
    BPNeuralNetworkTestProcess(nn, test_data)


if __name__ == '__main__':
    main()
