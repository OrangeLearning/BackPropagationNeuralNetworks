"""
    this file is to define the class for NN

    author : GeekVitaminC
"""
from BPNeuralNetworkTool import sigmoid
from BPNeuralNetworkTool import sigmoid_prime
import numpy as np
import time


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
        # for bias in self.biases:
        #     print(bias.shape)
        # for weight in self.weights:
        #     print(weight.shape)

        print(self.shapes)

    def frontFeedward(self, input):
        if len(input) != self.shapes[0]:
            raise ValueError("input size is not correct")

        res = np.array(input).reshape((len(input), 1))

        for b, w in zip(self.biases, self.weights):
            y = np.dot(w, res) + b
            print("y.shape = ", y.shape)
            res = sigmoid(y)

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
        # print("biases size = ", len(self.biases))
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            # print("z = ", z.shape)
            activations.append(activation)

        # print("activations = ", activations)

        # y^(1 - y^)(y - y^)
        delta = loss(activations[-1], y) * sigmoid_prime(zs[-1])
        # print("activation : ", activations[-2])
        # print("delta : ", delta)
        # time.sleep(1)
        nabla_b[-1] = delta

        # print("tt = ", activations[-2].T.shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # print(nabla_w[-1].shape)
        # time.sleep(1)
        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def updateStep(self, data: list, eta: float, loss):
        print("in update step function")
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in data:
            x = np.array(x)
            y = np.array(y)
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            print("x ,y = ", x.shape, ' ', y.shape)
            delta_nabla_b, delta_nabla_w = self.backPropagate(x, y, loss)
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
                cnt : the cnt of correct answer
        """
        test_results = []
        for x, y in data:
            res = self.frontFeedward(x)
            print("res = ", res, " y = ", y)
            test_results.append((np.argmax(res), y))

        cnt = 0
        for item_x , item_y in test_results:
            cnt += int(item_y[item_x] == 1)
        return test_results , cnt


def main():
    # nn = NeuralNetwork([4, 5, 3])

    a = np.array([3])
    b = np.array([1, 2, 3, 4, 5, 6])

    print(a.shape)
    print(b.shape)
    print(a * b.T)
    print((a * b.T).shape)
    print(np.dot(a, b))


if __name__ == '__main__':
    main()
