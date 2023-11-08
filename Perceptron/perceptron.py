# Machine Learning HW 3

import csv

# initial setup to find data filepath
from sys import path
from os.path import dirname, realpath

dir_path = dirname(realpath(__file__))
path.append(dir_path)
dir_parent = dirname(path[0])
path.append(dir_parent)

#import libraries
import numpy
import random
from random import shuffle
import copy
from copy import deepcopy

# define class Perceptron
class Perceptron:
    
    # constructor 
    def __init__(self, values, weight=1):
        self.attributes = numpy.array(values[0: len(values)-1])
        self.attributes = numpy.append(self.attributes, [1])
        self.label = -1 if values[-1] == 0 else 1
        self.weight = weight

# define file path
def examples_from_file(filename):
    examples = list()
    with open(filename, 'r') as train_data:
        for line in train_data:
            values = line.strip().split(',')
            for idx in range(len(values)):
                try:
                    values[idx] = float(values[idx])
                except ValueError:
                    print("check data here.")
            examples.append(Perceptron(values))
    return examples

if __name__ == '__main__':
    train_file = dir_path + "/bank-note/train.csv"
    test_file = dir_path + "/bank-note/test.csv"

    train = examples_from_file(train_file)
    test = examples_from_file(test_file)

    r = 0.8
    T = 10



# here we define the standard predict
def std_predict(samples, w):
    predicted = numpy.dot(w, samples.attributes)
    return -1 if predicted < 0 else 1


# define standard perceptron -- part a
def standard(epochs, T, r):
    w = numpy.zeros(len(epochs[0].attributes))
    for _ in range(T):
        shuffle(epochs)
        for samples in epochs:
            if std_predict(samples, w) != samples.label:
                w = w +  r * samples.attributes * samples.label
    return w


# define voted predict
def voted_predict(samples, w_list, correct):
    votes = 0
    for j in range(len(w_list)):
        decision = -1 if numpy.dot( w_list[j], samples.attributes) < 0 else 1
        votes += correct[j] * decision
    return -1 if votes < 0 else 1

# define voted perceptron -- part b
def vote(epochs, T, r):
    w = numpy.zeros(len(epochs[0].attributes))
    w_list = list()
    correct_list = list()
    amt_correct = 0
    for _ in range(T):
        shuffle(epochs)
        for samples in epochs:
            if std_predict(samples, w) != samples.label:
                w_list.append(deepcopy(w))
                correct_list.append(amt_correct)
                w += r * samples.attributes * samples.label
                amt_correct = 1
            else:
                amt_correct += 1
    return w_list, correct_list



# define average perceptron -- part c
def average(epochs, T, r):
    w = numpy.zeros(len(epochs[0].attributes))
    average = numpy.zeros(len(epochs[0].attributes))
    amt_correct = 0
    for _ in range(T):
        shuffle(epochs)
        for samples in epochs:
            if std_predict(samples, w) != samples.label:
                w += r * samples.attributes * samples.label
            else:
                average += w
    return w


def err_avg(epochs, marker):
    num_correct = 0
    num_incorrect = 0
    num_total = 0
    for samples in epochs:
        if marker(samples) * samples.label <= 0:
            num_incorrect += 1
        else:
            num_correct += 1  
        num_total += 1
    return num_incorrect / num_total



standard_weight_vector = standard(train, T, r)
# standard error train
standard_error_train = err_avg(train, lambda samples: std_predict(samples, standard_weight_vector))
# standard error test
standard_error_test = err_avg(test, lambda samples: std_predict(samples, standard_weight_vector))
voted_weight_vector, votes = vote(train, T, r)
# voted error test
voted_error_test = err_avg(test, lambda samples: voted_predict(samples, voted_weight_vector, votes))
# voted error train
voted_error_train = err_avg(train, lambda samples: voted_predict(samples, voted_weight_vector, votes))
# average weight
avgerage_weight = average(test, T, r)
# average test error
average_test_error = err_avg(test, lambda samples : std_predict(samples, avgerage_weight))
# average train error
average_train_error = err_avg(train, lambda samples : std_predict(samples, avgerage_weight))


# print allllll the things
print("The r vaule is: ", r)

# Question 2a
print("For Qestion 2a")
print("The standard preceptron method error on test data: ", standard_error_test)
print("The standard preceptron method weight vector is: ", standard_weight_vector)

#############################
# Question 2b
print("For Question 2b")
print("The voted preceptron method error on test data is: ", voted_error_test)


weight_vector = 0
for weight_vector in range(0, len(voted_weight_vector), 1) :
    print(weight_vector, voted_weight_vector[weight_vector], votes[weight_vector])
    
print("The voted total counts is: ", len(voted_weight_vector))

###########################

# Question 2c
print("For Question 2c")
print("The average preceptron method error on the test data is: ", average_test_error)
print("The average method preceptron weight vector is: ", avgerage_weight)






    

    
    
    

