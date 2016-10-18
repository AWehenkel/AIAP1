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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from scipy.stats import norm



class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):

    '''
    Attributes
    ----------
    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class
    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class
    classes_ : array, shape(n_classes)
        the value of the different classes
    prior_ : array, shape(n_classes)
        the likelihood of different classes
    '''

    def fit(self, X, y):
        """Fit a Gaussian naive Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================

        #For each type of output, compute the mean and the variance of the corresponding inputs
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        n_samples = X.shape[0]
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.prior_ = np.zeros(n_classes)
        i = 0
        for y_i in self.classes_:
            #Get all the input values for which the input is y_i
            X_i = X[y == y_i, :]
            #Compute the means and the variance for that output
            self.theta_[i] = np.mean(X_i, axis=0)
            self.sigma_[i] = np.var(X_i, axis=0)
            self.prior_[i] = float(X_i.shape[0])/float(n_samples)
            i += 1

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================
        #Return for each sample the number of the class for which the prob. was maximum
        prob = self.predict_proba(X)
        y = []
        for i in range(X.shape[0]):
            y.append(np.argmax(prob[i,:]))
        return np.array(y).T

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================
        n_classes = self.classes_.shape[0]
        n_samples = X.shape[0]
        p = np.zeros((n_samples, n_classes))
        for i in range(0, n_samples):
                p[i] = self.__predict_once_proba(X[i])

        return p

    def __gauss_cond_proba(self, x, y, xval):
        """Return conditional probability of the value xval of the feature x knowing the value y of the class.

                Parameters
                ----------
                x : int
                    The id of the feature.
                y : int
                    The id of the class.
                xval : int
                    The value of the feature.

                Returns
                -------
                p : float
                    The conditional probability
                """
        theta = self.theta_[y,x]
        sigma = self.sigma_[y,x]
        return norm(theta, sigma).pdf(xval)

    def __predict_once_proba(self, X):
        """Return probability estimates for the test data X.

                Parameters
                ----------
                X : array-like of shape = [n_features]
                    The input sample.

                Returns
                -------
                p : array of shape = [n_classes]
                    The class probabilities of the input sample. Classes are ordered
                    by lexicographic order.
                """
        n_classes = self.classes_.shape[0]
        p = np.zeros((n_classes))
        for cl in range(0, len(self.classes_)):
            p[cl] = self.prior_[cl]
            for fe in range(0, len(X)):
                p[cl] *= self.__gauss_cond_proba(fe, cl, X[fe])
        return p

if __name__ == "__main__":
    from data import make_data1
    from plot import plot_boundary
    from sklearn.naive_bayes import GaussianNB

    X1,y1 = make_data1(2000)
    training_set1 = [X1[0:149],y1[0:149]]
    test_set1 = [X1[150::],y1[150::]]
    neigh = GaussianNaiveBayes()
    neigh.fit(training_set1[0],training_set1[1])

    print(neigh.predict_proba(X1[150:152]))
    print(neigh.predict(X1[150:160]))
    neightest = GaussianNB()
    neightest.fit(training_set1[0],training_set1[1])
    print(neightest.predict_proba(X1[150:152]))
    print(neightest.predict(X1[150:160]))

    plot_boundary("NB_set1", neigh, test_set1[0], test_set1[1])
    plot_boundary("NB_set1_check", neightest, test_set1[0], test_set1[1])

