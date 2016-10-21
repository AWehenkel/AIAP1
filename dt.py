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
    '''This function builds a decision model on one training set
        and displays the decision boundary with the corresponding testing sets.

        Parameters
        ----------
        dt_max_depth  : int > 0
            maximum depth of the decision tree model

        data    :   list(X, y)
            Where
                X : array of shape [n_samples, nb_feature]
                The input samples.

                y : array of shape [n_samples]
                The output values.

        nb_training :   int > 0
            The number of sample that will be used for the training

        file_name   :   string
            The name of the file to register the plot.

        '''

    training = [data[0][:nb_training], data[1][:nb_training]]
    testing = [data[0][nb_training + 1:], data[1][nb_training + 1:]]
    dt = DecisionTreeClassifier(max_depth=dt_max_depth)
    dt.fit(training[0], training[1])
    plot_boundary(file_name, dt, testing[0], testing[1])

def trainAndComputeErrors(data, nb_training, dt_max_depth):
    '''This function builds a decision model on one training set
        and return testing error and training error with the corresponding testing sets.

        Parameters
        ----------
        dt_max_depth  : int > 0
            maximum depth of the decision tree model

        data    :   list(X, y)
            Where
                X : array of shape [n_samples, nb_feature]
                The input samples.

                y : array of shape [n_samples]
                The output values.

        nb_training :   int > 0
            The number of sample that will be used for the training

        Returns
        -------
        errors : list of the form = (training_error, testing_error)
            The percentage of error on the training set and on the testing set.
        '''
    training = [data[0][:nb_training], data[1][:nb_training]]
    testing = [data[0][nb_training + 1:], data[1][nb_training + 1:]]
    dt = DecisionTreeClassifier(max_depth=dt_max_depth)
    dt.fit(training[0], training[1])
    training_error = dt.score(training[0], training[1])
    testing_error = dt.score(testing[0], testing[1])
    return (training_error, testing_error)

def plotForQuestion3(data, nb_training, dt_max_depth, title):
    '''This function builds a decision model on one training set
        and plot the error curves(on training and testing) with value of max depth between 1 and dt_max_depth.

        Parameters
        ----------
        dt_max_depth  : int > 0
            maximum depth to compute the error.

        data    :   list(X, y)
            Where
                X : array of shape [n_samples, nb_feature]
                The input samples.

                y : array of shape [n_samples]
                The output values.

        nb_training :   int > 0
            The number of sample that will be used for the training

        title   :   string
            The name of the file to register the plot.

        '''
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

def trainAndCrossScore(data, dt_max_depth, nb_fold):
    '''This function computes the k-fold cross score for the decision tree.

        Parameters
        ----------
        dt_max_depth  : int > 0
            maximum depth of the decision tree model.

        data    :   list(X, y)
            Where
                X : array of shape [n_samples, nb_feature]
                The input samples.

                y : array of shape [n_samples]
                The output values.

        nb_fold :   int > 0
            The number of subset, number k.

        '''
    dt = DecisionTreeClassifier(max_depth=dt_max_depth)
    return cross_val_score(dt, data[0], data[1], cv=nb_fold)

def crossOptimize(data, nb_fold, dt_max_depth):
    '''This function computes the best max_depth parameter with a k-fold cross score optimization.

            Parameters
            ----------
            dt_max_depth  : int > 0
                maximum depth of the decision tree model.

            data    :   list(X, y)
                Where
                    X : array of shape [n_samples, nb_feature]
                    The input samples.

                    y : array of shape [n_samples]
                    The output values.

            nb_fold :   int > 0
                The number of subset, number k.

            '''
    max_v = [0, 0]
    for depth in range(1, dt_max_depth + 1):
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
    #crossOptimize(set2, 10, 45)



