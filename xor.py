# -*- coding: utf-8 -*-
# @Author : xushiqi
# @Email  : xushiqitc@163.com
# @Env    : Python 3.5
# @IDE    : PyCharm


import time
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivate(x):
    return 1. * (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))


def train(X, Y, w1, w2, w3, b1, b2, b3):
    # 前向传播
    A1 = np.dot(X, w1) + b1
    Z1 = relu(A1)

    A2 = np.dot(Z1, w2) + b2

    shortcut = A1 + A2
    Z2 = relu(shortcut)

    A3 = np.dot(Z2, w3) + b3
    y = sigmoid(A3)

    # 后向传播Loss = 0.5*(y-Y)^2
    dy = y - Y
    dA3 = sigmoid_derivate(A3) * dy    # dL/dA3 = (dL/dy) * (dy/dA3)
    dshortcut = relu_derivate(shortcut) * np.dot(dA3, w3.T)  # dL/dshortcut = (dl/dA3) * (dA3/dZ2) * (dZ2/dshortcut)
    dA1 = dshortcut  # dL/dA1 = (dL/dshortcut) * (dshortcut/dA1)
    dA2 = dshortcut  # dL/dA2 = (dL/dshortcut) * (dshortcut/dA2)
    dA1 += relu_derivate(A1) * np.dot(dA2, w2.T)

    dw3 = np.dot(Z2.T, dA3)
    dw2 = np.dot(Z1.T, dA2)
    dw1 = np.dot(X.T, dA1)

    loss = np.mean(-np.sum(Y * np.log(y) + (1 - Y) * np.log(1 - y), axis=1))

    return loss, (dw1, dw2, dw3, dA1.sum(axis=0), dA2.sum(axis=0), dA3.sum(axis=0))


if __name__ == '__main__':
    n_in = 10
    n_hidden1 = 10
    n_hidden2 = 10
    n_out = 10

    np.random.seed(2018)

    w1 = np.random.normal(scale=0.1, size=(n_in, n_hidden1))
    w2 = np.random.normal(scale=0.1, size=(n_hidden1, n_hidden2))
    w3 = np.random.normal(scale=0.1, size=(n_hidden2, n_out))

    b1 = np.zeros(n_hidden1)
    b2 = np.zeros(n_hidden2)
    b3 = np.zeros(n_out)

    n_samples = 300
    batch_size = 30

    learning_rate = 0.001
    momentum = 0.9

    X = np.random.binomial(1, 0.5, (n_samples, n_in))
    Y = X ^ 1
    params = [w1, w2, w3, b1, b2, b3]
    n_batches = X.shape[0] // batch_size
    for epoch in range(2000):
        err = []
        upd = [0] * len(params)
        t0 = time.clock()
        for i in range(n_batches):
            s = slice(batch_size * i, batch_size * (i + 1))
            loss, grad = train(X[s], Y[s], *params)

            for j in range(len(params)):
                params[j] -= upd[j]

            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]

            err.append(loss)
        print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (epoch, float(np.mean(err)), time.clock() - t0))
