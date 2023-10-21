# This is a least means squares method for linear regression task


import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import math

import numpy as np

def MSE(pred, target) -> float:
    assert len(pred) == len(target)
    pred, target = np.array(pred), np.array(target)
    return np.sum((target - pred)**2) / 2

def predict(weights, x) -> list:
        return np.dot(weights, x)
    
def batch_gradient_descent(x, y, r = 1, epochs = 10, threshold = 1e-6):

    # initialize weights
    w = np.ones_like(x[0])
    losses= []
    lastloss = 9999
    distance = 1
    # for T epochs...
    for ep in range(epochs):
        if distance <= threshold: break
        # compute gradient of J(w) at w^t
        delJ = np.zeros_like(w)

        for j in range(len(delJ)):
            for xi, yi in zip(x, y):
                delJ[j] -= (yi - np.dot(w,xi)) * xi[j]

        # update weights
        w = w - r * delJ

        # compute loss
        loss = 0
        for xi, yi in zip(x, y):
            loss += (yi - np.dot(w, xi))**2
        loss /= 2
        
        distance = abs(loss - lastloss)
        lastloss = loss
        losses.append(loss)

    print(f"converged at epoch {ep} to {distance}")
    return w, losses

def sgd(x, y, r= 1.0, epochs= 10, threshold = 1e-6):

    w = np.ones_like(x[0])

    losses= []
    lastloss = 9999
    distance = 1
    for ep in range(epochs):
        if distance <= threshold: break
        # for each element, update weights
        for xi, yi in zip(x, y):
            for j in range(len(w)):
                w[j] += r * (yi - np.dot(w, xi)) * xi[j]

            # compute loss
            loss = 0
            for xi, yi in zip(x, y):
                loss += (yi - np.dot(w, xi))**2
            loss /= 2
            
            distance = abs(loss - lastloss)
            lastloss = loss
            losses.append(loss)

    print(f"converged at epoch {ep} to {distance}")
    return w, losses



if __name__=="__main__":
    
    x = []
    y = []
    
    # with open(Path("concrete", "train.csv")) as csv_file:
    with open(Path("concrete", "train.csv")) as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            x.append([ float(ex) for ex in sample[:-1]])
            y.append(float(sample[-1]))
    
    x_test = []
    y_test = []
    
    # with open(Path("concrete", "train.csv")) as csv_file:
    with open(Path("concrete", "train.csv")) as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            x_test.append([ float(ex) for ex in sample[:-1]])
            y_test.append(float(sample[-1]))

    w = None
    losses = None
    r_step  = [1.0, 0.5, 0.25, 0.125, 0.0625]
    for r in r_step:
        w, losses = batch_gradient_descent(x, y, r)
        w_sgd, losses_sgd = sgd(x, y, r)

    # compute loss
    loss = 0
    for xi, yi in zip(x_test, y_test):
        loss += (yi - np.dot(w, xi))**2
    loss /= 2

    print(f"Cost for batch gradient desent test using w: {loss}")

    fig, ax = plt.subplots(2,1)
    ax = ax.ravel()
    ax[0].plot(losses, range(len(losses)), color = 'tab:blue', label = "batch")
    ax[1].plot(losses_sgd, range(len(losses_sgd)), color = 'tab:orange', label = "stochastic")

    plt.savefig("gd_cost.png")
    plt.clf()


