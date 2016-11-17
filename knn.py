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
from sklearn.neighbors import KNeighborsClassifier


# (Question 2)
from plot import plot_boundary
from sklearn.model_selection import cross_val_score
from matplotlib import patches
# Put your functions here

# ----- Question 2.1 ------ #

def question21(n_neighbours, X1, y1, X2, y2, title=''):
    '''This function builds a k-nearest neighbours model on two training sets
    and displays the decision boundary with the corresponding testing sets.

    Parameters
    ----------
    n_neighbours  : int > 0
        number of neighbours used in the k-nearest neighbours model

    X1, X2:
        input values of the two datasets

    y1, y2:
        output values of the two datasets

    title    : string, (default = '')
        title given to the plots

    '''

    #Makes the computation for the first dataset
    training_set1 = [X1[0:149],y1[0:149]]
    test_set1 = [X1[150::],y1[150::]]
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(training_set1[0],training_set1[1])
    plot_boundary(title+"_set1", neigh, test_set1[0], test_set1[1])

    #Makes the computation for the second dataset
    training_set2 = [X2[0:149],y2[0:149]]
    test_set2 = [X2[150::],y2[150::]]
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(training_set2[0],training_set2[1])
    plot_boundary(title+"_set2", neigh, test_set2[0], test_set2[1])



# ----- Question 2.2 ------ #

def question22(n_min, n_max, n_step, X1, y1, X2, y2):
    '''This function builds a k-nearest neighbours model on two training sets
    for several values of n_neighbours and displays the decision boundary with
    the corresponding testing sets.

    Parameters
    ----------
    n_min : int > 0
        the minimal number of neighbours for which the model must be computed

    n_max : int >= n_min
        the maximal number of neighbours for which the model must be computed

    n_step: int > 0
        step between two successive n_neighbours that you test

    X1, X2:
        input values of the two datasets

    y1, y2:
        output values of the two datasets

    '''

    while n_min <= n_max :
        question21(n_min, X1, y1, X2, y2, "test_"+ str(n_min) +"neigh")
        n_min += n_step



# ----- Question 2.3 ------ #

def plotQuestion23(n_min, n_max, accu_train, accu_test, title):
    '''This function plots the accuracy on the learning and training sets for various values of
        the n_neighbours parameter.

        Parameters
        ----------
        n_min : int > 0
        the minimal number of neighbours for which the model must be computed

        n_max : int >= n_min
            the maximal number of neighbours for which the model must be computed

        accu_train: vector of float
            accuracy on the learning sets for different values of n_neighbours

        accu_test: vector of float
            accuracy on the testing sets for different values of n_neighbours

        title: string
            title of the graph
        '''

    #Changing accuracy to error ratio
    error_train = [1-i for i in accu_train]
    error_test = [1-i for i in accu_test]

    plt.plot(range(n_min, n_max+1), error_train, 'r', range(n_min, n_max+1), error_test, 'b')
    plt.xlabel("N_neighbours")
    plt.ylabel("Error ratio")
    r_patch = patches.Patch(color='r', label='Training error')
    b_patch = patches.Patch(color='b', label='Testing error')
    plt.legend(handles=[r_patch, b_patch])
    plt.title(title)
    plt.show()

def computeAccu(n_min, n_max, n_iter, make_data):
    '''This function computes the accuracy of the k-neighbours model on the
    learning and testing sets for different number of neighbours

    Parameters
    ----------
    n_min : int > 0
        the minimal number of neighbours for which the model must be computed

    n_max : int >= n_min
        the maximal number of neighbours for which the model must be computed

    n_iter: int > 0
        number of iterations

    make_data:
        function creating a dataset

    Returns
    -------
    accu_train: vector of float
        accuracy on the learning sets for different values of n_neighbours

    accu_test: vector of float
        accuracy on the testing sets for different values of n_neighbours

    '''

    if n_max >= n_min:
        accu_train = [0 for x in range(n_max-n_min+1)]
        accu_test = [0 for x in range(n_max-n_min+1)]

        for i in range(n_iter):
            #Create the dataset
            X,y = make_data(2000)
            training_set = [X[0:149],y[0:149]]
            test_set = [X[150::],y[150::]]

            for i in range(n_max-n_min+1):
                #Compute accuracy for the dataset
                neigh = KNeighborsClassifier(n_neighbors=i+n_min)
                neigh.fit(training_set[0],training_set[1])
                accu_train[i] += neigh.score(training_set[0], training_set[1])
                accu_test[i] += neigh.score(test_set[0], test_set[1])

        #Compute the mean of the results
        accu_train = [i/n_iter for i in accu_train]
        accu_test = [i/n_iter for i in accu_test]

        return [accu_train, accu_test]

def question23(n_min, n_max, n_iter):
    '''This function computes the accuracy of the k-neighbours model on the
    learning and testing sets for different number of neighbours and plot the corresponding
    error curves. It does this for the two data sets.

    Parameters
    ----------
    n_min : int > 0
        the minimal number of neighbours for which the model must be computed

    n_max : int >= n_min
        the maximal number of neighbours for which the model must be computed

    n_iter: int > 0
        number of iterations

    '''

    #DataSet1
    [accu_train1, accu_test1] = computeAccu(n_min, n_max, n_iter, make_data1)
    plotQuestion23(n_min, n_max, accu_train1, accu_test1, "Dataset1")

    #DataSet2
    [accu_train2, accu_test2] = computeAccu(n_min, n_max, n_iter, make_data2)
    plotQuestion23(n_min, n_max, accu_train2, accu_test2, "Dataset2")



# ----- Question 2.4 ------ #

def histQuestion24(data_name, best_neighbours):
    '''This function plots an histogram of the values contained in the vector
    best_neighbours

    Parameters
    ----------
    data_name: string
       name of the data set on which the cross validation algorithm was used

    best_neighbours: np.matrix
        this vector contains the number of neighbours that optimized a cross validation algorithm
    '''

    plt.hist(best_neighbours, bins=20)
    plt.xlabel('Optimal numbers of neighbours')
    plt.ylabel('Number of occurences')
    plt.title("Histogram of best neighbours for " + data_name + " with " + str(len(best_neighbours)) + " iterations")
    plt.show()


def optimize(n_max, make_data, nb_fold):
    '''This function uses a k-fold cross validation strategy to optimize the value of
        the n_neighbors parameter.

        Parameters
        ---------
        n_max: int > 0
            the range of n_neighbours parameters that are going to be tested

        make_data:
            function to create a data set

        nb_fold: int > 0
            number of folds used in the cross validation strategy

        Return
        ------
        best_score: float > 0, <= 1
            best score obtained with the ten-fold cross validation strategy

        best_neigh: > 0, <= n_max
            value of n_neighbours for which the ten-fold cross validation strategy was optimal
    '''

    best_score = 0
    best_neigh = 1
    X,y = make_data(2000)

    for i in range(1, n_max+1):
        neigh = KNeighborsClassifier(n_neighbors=i)
        score = sum(cross_val_score(neigh, X, y, cv=nb_fold))/nb_fold

        #Update the return values
        if score > best_score:
            best_score = score
            best_neigh = i

    return [best_score, best_neigh]

def question24(n_max, n_iter):
    '''This function uses a ten-fold cross validation strategy to optimize the value of
        the n_neighbors parameter for two datasets.

        Parameters
        ----------
        n_max: int > 0
            the range of n_neighbours parameters that are going to be tested

        n_iter: int > 0
            numbers of iterations

        Return
        ------
        best_score1, best_score2: float > 0, <= 1
            best scores for each data sets obtained with the ten-fold cross validation strategy

        best_neigh1, best_neigh2: int > 0, <= n_max
            value of n_neighbours for each data sets for which the ten-fold cross validation strategy was optimal

        '''

    nb_fold = 10
    best_score1 = np.zeros(n_iter)
    best_score2 = np.zeros(n_iter)
    best_neigh1 = np.zeros(n_iter)
    best_neigh2 = np.zeros(n_iter)
    for i in range(n_iter):
        [best_score1[i], best_neigh1[i]] = optimize(n_max, make_data1, nb_fold)
        [best_score2[i], best_neigh2[i]] = optimize(n_max, make_data2, nb_fold)

    #Plot the results
    histQuestion24("DataSet1", best_neigh1)
    mean1 = np.average(best_neigh1)
    std1 = np.std(best_neigh1)
    print("The mean is " + str(mean1) + " and the std is " + str(std1) + " for data set 1")
    histQuestion24("DataSet2", best_neigh2)
    mean2 = np.average(best_neigh2)
    std2 = np.std(best_neigh2)
    print("The mean is " + str(mean2) + " and the std is " + str(std2) + " for data set 2")
    return [best_score1, best_neigh1, best_score2, best_neigh2]

if __name__ == "__main__":

    #Tests
    X1,y1 = make_data1(2000)
    X2,y2 = make_data2(2000)

    print(X2[0:149])
    '''
    #Question 2.1
    question21(5, X1, y1, X2, y2, "test21")

    #Question 2.2
    question22(1, 9, 1, X1, y1, X2, y2)
    question22(10, 100, 10, X1, y1, X2, y2)

    #Question 2.3
    question23(1, 149, 149)

    #Question 2.4
    score = question24(50, 100)
    print(np.mean(score[0]), np.std(score[0]), np.mean(score[2]), np.std(score[2]))
    '''