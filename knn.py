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
# Put your functions here

def question21(n_neighbours, train_size=150, train_pos=0, title=''):
    '''This function builds a k-nearest neighbours model on two training sets
    and displays the decision boundary with the corresponding testing sets.

    Parameters
    ----------
    n_neighbours  : int > 0
        number of neighbours used in the k-nearest neighbours model

    title    : string, (default = '')
        title given to the plots

    '''

    #Makes the computation for the first dataset
    X,y = make_data1(2000)
    training_set = [X[0:149],y[0:149]]
    test_set = [X[150::],y[150::]]
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(training_set[0],training_set[1])
    plot_boundary(title+"_set1", neigh, test_set[0], test_set[1])

    #Makes the computation for the second dataset
    X,y = make_data2(2000)
    training_set = [X[0:149],y[0:149]]
    test_set = [X[150::],y[150::]]
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(training_set[0],training_set[1])
    plot_boundary(title+"_set2", neigh, test_set[0], test_set[1])



def question22(n_min, n_max):
    '''This function builds a k-nearest neighbours model on two training sets
    for several values of n_neighbours and displays the decision boundary with
    the corresponding testing sets.

    Parameters
    ----------
    n_min : int > 0
        the minimal number of neighbours for which the model must be computed

    n_max : int >= n_min
        the maximal number of neighbours for which the model must be computed

    '''

    while n_min <= n_max :
        question21(n_min, "test_"+ str(n_min) +"neigh")
        n_min += 1


#!!! Pour cette question, il faudrait peut etre faire plusieurs itérations pour avoir une meilleure estimation
def question23(n_min, n_max):
    '''This function computes the accuracy of the k-neighbours model on the
    learning and testing sets for different number of neighbours and plot the corresponding
    error curves. It does this for the two data sets.

    Parameters
    ----------
    n_min : int > 0
        the minimal number of neighbours for which the model must be computed

    n_max : int >= n_min
        the maximal number of neighbours for which the model must be computed

    '''

    if n_max >= n_min:
        accu1_train = [0 for x in range(n_max-n_min+1)]
        accu1_test = [0 for x in range(n_max-n_min+1)]
        accu2_train = [0 for x in range(n_max-n_min+1)]
        accu2_test = [0 for x in range(n_max-n_min+1)]

        #Create dataset 1
        X,y = make_data1(2000)
        training_set1 = [X[0:149],y[0:149]]
        test_set1 = [X[150::],y[150::]]

        #Create dataset 2
        X,y = make_data2(2000)
        training_set2 = [X[0:149],y[0:149]]
        test_set2 = [X[150::],y[150::]]

        for i in range(n_max-n_min+1):
            #Compute accuracy for dataset 1
            neigh1 = KNeighborsClassifier(n_neighbors=i+n_min)
            neigh1.fit(training_set1[0],training_set1[1])
            accu1_train[i] = neigh1.score(training_set1[0], training_set1[1])#!!!! PAS SUR DU TOUT QU'IL FAUT UTILISER CETTE FONCTION LA
            accu1_test[i] = neigh1.score(test_set1[0], test_set1[1])
            #Compute accuracy for dataset 2
            neigh2 = KNeighborsClassifier(n_neighbors=i+n_min)
            neigh2.fit(training_set2[0],training_set2[1])
            accu2_train[i] = neigh1.score(training_set2[0], training_set2[1])
            accu2_test[i] = neigh1.score(test_set2[0], test_set2[1])

        #Plot
        plt.plot(accu1_train)
        plt.plot(accu1_test)
        plt.plot(accu2_train)
        plt.plot(accu2_test)
        plt.show()

def question24():



if __name__ == "__main__":

    question23(1,50)
