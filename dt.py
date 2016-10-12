"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from data import make_data1, make_data2
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from matplotlib import patches
from sklearn.model_selection import cross_val_score

from plot import plot_boundary


# (Question 1)

def trainAndPlot(data, nb_training, dt_max_depth, file_name = "default"):
    training = [data[0][:nb_training], data[1][:nb_training]]
    testing = [data[0][nb_training + 1:], data[1][nb_training + 1:]]
    dt = DecisionTreeClassifier(max_depth=dt_max_depth)
    dt.fit(training[0], training[1])
    plot_boundary(file_name, dt, testing[0], testing[1])

def trainAndComputeErrors(data, nb_training, dt_max_depth):
    training = [data[0][:nb_training], data[1][:nb_training]]
    testing = [data[0][nb_training + 1:], data[1][nb_training + 1:]]
    dt = DecisionTreeClassifier(max_depth=dt_max_depth)
    dt.fit(training[0], training[1])
    training_error = dt.score(training[0], training[1])
    testing_error = dt.score(testing[0], testing[1])
    return (training_error, testing_error)

def plotForQuestion3(data, nb_training, dt_max_depth, title):
    training_error = []
    testing_error = []
    for depth in range(1, dt_max_depth):
        error1 = 0.0
        error2 = 0.0
        nb_set = len(data)
        for i in range(0, nb_set):
            errors = trainAndComputeErrors(data[i], nb_training, depth)
            error1 += errors[0]/nb_set
            error2 += errors[1]/nb_set
        training_error.append(1 - error1)
        testing_error.append(1 - error2)
    pyplot.plot(range(1, dt_max_depth), training_error, 'r', range(1, dt_max_depth), testing_error, 'b')
    pyplot.xlabel("Max depth")
    pyplot.ylabel("Error ratio")
    r_patch = patches.Patch(color='r', label='Training errors')
    b_patch = patches.Patch(color='b', label='Testing errors')
    pyplot.legend(handles=[r_patch, b_patch])
    pyplot.title(title)
    pyplot.show()

def trainAndCrossScore(data, max_depth, nb_fold):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    return cross_val_score(dt, data[0], data[1], cv=nb_fold)

def crossOptimize(data, nb_fold, max_depth):
    max_v = [0, 0]
    for depth in range(1, max_depth + 1):
        result = trainAndCrossScore(data, depth, nb_fold)
        result = sum(result) / len(result)
        if (result > max_v[0]):
            max_v[0] = result
            max_v[1] = depth

    print("Best result = %f for the depth %d" % (max_v[0], max_v[1]))

if __name__ == "__main__":

    #Data init
    set1 = make_data1(2000)
    set2 = make_data2(2000)

    #Tests Q1
    #trainAndPlot(set1, 150, None, "set1")
    #trainAndPlot(set2, 150, None, "set2")

    #Tests Q2
    #for depth in range(1, 10):
        #trainAndPlot(set1, 150, depth, ("set1_depth%d" % depth))
        #trainAndPlot(set2, 150, depth, ("set2_depth%d" % depth))

    #Tests Q3
    #sets1 = []
    #sets2 = []
    #for i in range(0, 10):
        #sets1.append(make_data1(2000))
        #sets2.append(make_data2(2000))
    #plotForQuestion3(sets1, 150, 15, "dataset 1")
    #plotForQuestion3(sets2, 150, 15, "dataset 2")

    #Tests Q4
    #crossOptimize(set1, 10, 25)
    crossOptimize(set2, 10, 45)



