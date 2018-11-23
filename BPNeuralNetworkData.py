"""
    This file is to provide data
"""
import random
import time

IrisAddress = "/home/orangeluyao/Data/data/IrisData/iris.data"


def getIrisData():
    res = []
    yLabel = {}
    rLabel = {}

    with open(IrisAddress, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

        for line in lines:
            sList = str(line).strip().split(',')

            if sList[-1] not in yLabel.keys():
                yLabel[sList[-1]] = len(yLabel.keys())
                rLabel[yLabel[sList[-1]]] = sList[-1]

        y_cnt = len(yLabel.keys())

        for i in range(len(lines)):
            sList = str(lines[i]).strip().split(',')
            ys = [0 for i in range(y_cnt)]
            index = yLabel[sList[-1]]
            ys[index] = 1
            xs = [float(s) for s in sList[:-1]]

            res.append((xs, ys))

    # print(res)
    return res, yLabel, rLabel


def divide_train_test(data: list, rate: float):
    """

        :param data: data list
        :param rate: data rate
        :return:
    """

    train_data = []
    test_data = []

    random.seed(time.time())
    for item in data:
        if random.random() < rate:
            train_data.append(item)
        else:
            test_data.append(item)

    return train_data, test_data


def main():
    res, y, r = getIrisData()
    print(res)
    print(y)
    print(r)

    ll = range(2, 3)
    for i in ll:
        print(i)


if __name__ == '__main__':
    main()
