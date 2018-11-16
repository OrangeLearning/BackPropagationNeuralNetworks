"""
    this file is to define the class for NN

    author : GeekVitaminC
"""
from BPNeuralNetworkTool import sigmoid
from BPNeuralNetworkTool import sigmoid_prime
import numpy as np


class NeuralNetwork:
    def __init__(self, shapes):
        """
        :param shapes: shapes is a list or tuple which represent the shape of every layer including input and output

        """
        self.layers = len(shapes)
        self.shapes = shapes

        # add a bias besides input layer
        self.biases = [np.random.randn(y, 1) for y in shapes[1:]]
        self.weights = [
            # Good code style from Michael Nielsen !!!
            np.random.randn(y, x) for (x, y) in zip(shapes[:-1], shapes[1:])
        ]

    def feedward(self, input):
        if len(input) != self.shapes[0]:
            raise ValueError("input size is not correct")

        res = input

        for b, w in zip(self.biases, self.weights):
            res = sigmoid(np.dot(w, res) + b)

        return res

    def backPropagate(self, x, y, loss):
        """
            :param x:
            :param y:
            :param loss: the loss function
            :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        # calc the activation result
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        activations = np.array(activations)

        delta = loss(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # transpose() means X.T
        nabla_w[-1] = np.dot(delta , activation[-2].transpose())

        for l in range(2,self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def updateStep(self, data : list, eta : float, loss):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in data:
            delta_nabla_b, delta_nabla_w = self.backPropagate(x, y,loss)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(data)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(data)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, data):
        """
            :param data:
            :return:
                test_results : the result of the output
                cnt : the cnt of same result
        """
        test_results = []
        for x, y in data:
            res = self.feedward(x)
            test_results.append((np.argmax(res), y))

        cnt = 0
        for y0, y in test_results:
            cnt += int(y0 == y)

        return test_results, cnt


def main():
    nn = NeuralNetwork([2, 3, 4])


if __name__ == '__main__':
    main()
