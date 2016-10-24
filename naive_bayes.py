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
        proba = 1/np.sqrt(2*np.pi*sigma)*np.exp(-((xval - theta)**2)/(2*sigma))
        #return norm(theta, sigma).pdf(xval)
        return proba

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
        return p/p.sum()

if __name__ == "__main__":
    from data import make_data1
    from data import make_data2
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    test_precision = 0
    training_precision = 0
    for i in range(500):
        #Create the data set
        X1,y1 = make_data2(2000) #Choose which data set you use
        training_set1 = [X1[:149],y1[:149]]
        test_set1 = [X1[150:],y1[150:]]
        neigh = GaussianNaiveBayes()
        neigh.fit(training_set1[0],training_set1[1])

        #Compute the precision
        test_prediction = neigh.predict(X1[150:])
        test_precision += 1.0 - float(np.absolute(y1[150:] - test_prediction).sum())/1850.0
        training_prediction = neigh.predict(X1[:149])
        training_precision += 1.0 - float(np.absolute(y1[:149] - training_prediction).sum())/150.0
    print(test_precision/500)
    print(training_precision/500)

    #Gaussian display
    sigma1 = neigh.sigma_
    sigma2 = sigma1
    x = np.linspace(-3, 3, 1000)
    for i in range(2):
        sigma1[0][i] = neigh.sigma_[0][i]
        sigma1[1][i] = neigh.sigma_[1][i]
        gauss1 = mlab.normpdf(x, neigh.theta_[0][i], sigma1[0][i])
        gauss2 = mlab.normpdf(x, neigh.theta_[1][i], sigma2[1][i])
        twodgauss = np.zeros((len(gauss1), len(gauss1)), dtype=np.float)
        for d1 in range(len(gauss1)):
            for d2 in range(len(gauss2)):
                twodgauss[d1][d2] = gauss1[d1]*gauss2[d2]
        CS = plt.contour(x, x, twodgauss)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.title('Data set 2 - Feature ' + str(i+1))
        plt.xlabel('X_0')
        plt.ylabel('X_1')
        plt.show()





    #plot_boundary("NB_set2", neigh, test_set1[0], test_set1[1])
    #plot_boundary("NB_set2_check", neightest, test_set1[0], test_set1[1])

