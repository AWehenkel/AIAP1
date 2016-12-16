# Daniel Alabi and Cody Wang
# ======================================
# SvdMatrix:
# generates matrices U and V such that
# U * V^T closely approximates
# the original matrix (in this case, the utility
# matrix M)
# =======================================


import math
import random
import pandas as pd
import time
import numpy as np
import CollaborativeFiltering2

"""
Rating class.
Store every rating associated with a particular
userid and movieid.
================Optimization======================
"""


class Rating:
    def __init__(self, userid, movieid, rating):
        # to accomodate zero-indexing for matrices
        self.uid = userid - 1
        self.mid = movieid - 1

        self.rat = rating


class SvdMatrix:
    """
    trainfile -> name of file to train data against
    nusers -> number of users in dataset
    nmovies -> number of movies in dataset
    r -> rank of approximation (for U and V)
    lrate -> learning rate
    regularizer -> regularizer
    typefile -> 0 if for smaller MovieLens dataset
                1 if for medium or larger MovieLens dataset
    """

    def __init__(self, trainfile, nusers, nmovies, r=30, lrate=0.035, regularizer=0.01, typefile=0):
        self.trainrats = []
        self.testrats = []

        self.nusers = nusers
        self.nmovies = nmovies

        if typefile == 0:
            self.readtrainsmaller(trainfile)
        elif typefile == 1:
            self.readtrainlarger(trainfile)

        # get average rating
        avg = self.averagerating()
        # set initial values in U, V using square root
        # of average/rank
        initval = math.sqrt(avg / r)

        # U matrix
        self.U = [[initval] * r for i in range(nusers)]
        print(len(self.U))

        # V matrix -- easier to store and compute than V^T
        self.V = [[initval] * r for i in range(nmovies)]

        self.r = r
        self.lrate = lrate
        self.regularizer = regularizer
        self.minimprov = 0.001
        self.maxepochs = 30

    """
    Returns the dot product of v1 and v2
    """

    def dotproduct(self, v1, v2):
        return sum([v1[i] * v2[i] for i in range(len(v1))])

    """
    Returns the estimated rating corresponding to userid for movieid
    Ensures returns rating is in range [1,5]
    """

    def calcrating(self, uid, mid):
        p = self.dotproduct(self.U[uid], self.V[mid])
        if p > 5:
            p = 5
        elif p < 1:
            p = 1
        return p

    """
    Returns the average rating of the entire dataset
    """

    def averagerating(self):
        avg = 0
        n = 0
        for i in range(len(self.trainrats)):
            avg += self.trainrats[i].rat
            n += 1
        return float(avg / n)

    """
    Predicts the estimated rating for user with id i
    for movie with id j
    """

    def predict(self, i, j):
        return self.calcrating(i, j)

    """
    Trains the kth column in U and the kth row in
    V^T
    See docs for more details.
    """

    def train(self, k):
        sse = 0.0
        n = 0
        for i in range(len(self.trainrats)):
            # get current rating
            crating = self.trainrats[i]
            err = crating.rat - self.predict(crating.uid, crating.mid)
            sse += err ** 2
            n += 1

            uTemp = self.U[crating.uid][k]
            vTemp = self.V[crating.mid][k]

            self.U[crating.uid][k] += self.lrate * (err * vTemp - self.regularizer * uTemp)
            self.V[crating.mid][k] += self.lrate * (err * uTemp - self.regularizer * vTemp)
        return math.sqrt(sse / n)

    """
    Trains the entire U matrix and the entire V (and V^T) matrix
    """

    def trainratings(self):
        # stub -- initial train error
        oldtrainerr = 1000000.0
        print("ok")
        for k in range(self.r):
            print("k=", k)
            for epoch in range(self.maxepochs):
                trainerr = self.train(k)

                # check if train error is still changing
                if abs(oldtrainerr - trainerr) < self.minimprov:
                    break
                oldtrainerr = trainerr
                print("epoch=", epoch, "; trainerr=", trainerr)

    """
    Calculates the RMSE using between arr
    and the estimated values in (U * V^T)
    """

    def calcrmse(self, arr):
        nusers = self.nusers
        nmovies = self.nmovies
        sse = 0.0
        total = 0
        for i in range(len(arr)):
            crating = arr[i]
            sse += (crating.rat - self.calcrating(crating.uid, crating.mid)) ** 2
            total += 1
        return math.sqrt(sse / total)

    """
    Read in the ratings from fname and put in arr
    Use splitter as delimiter in fname
    """

    def readinratings(self, fname, arr, splitter="\t"):
        f = open(fname)

        for line in f:
            newline = [int(each) for each in line.split(splitter)]
            userid, movieid, rating = newline[0], newline[1], newline[2]
            arr.append(Rating(userid, movieid, rating))

        arr = sorted(arr, key=lambda rating: (rating.uid, rating.mid))
        return len(arr)

    """
    Read in the smaller train dataset
    """

    def readtrainsmaller(self, fname):
        return self.readinratings(fname, self.trainrats, splitter="\t")

    """
    Read in the large train dataset
    """

    def readtrainlarger(self, fname):
        return self.readinratings(fname, self.trainrats, splitter="::")

    """
    Read in the smaller test dataset
    """

    def readtestsmaller(self, fname):
        return self.readinratings(fname, self.testrats, splitter="\t")

    """
    Read in the larger test dataset
    """

    def readtestlarger(self, fname):
        return self.readinratings(fname, self.testrats, splitter="::")


if __name__ == "__main__":
    # ========= test SvdMatrix class on smallest MovieLENS dataset =========
    init = time.time()
    output = pd.read_csv('data/output_train.csv', delimiter=',').values
    train = pd.read_csv('data/data_train.csv', delimiter=',').values
    train = np.append(train, output, axis=1)
    np.random.shuffle(train)
    nb_test = 15000
    '''
    with open("aggregated_output_ids.base", 'w') as f:
        for i in range(train.shape[0] - nb_test):
            line = "%d\t%d\t%d\n" % (train[i, 0], train[i, 1], train[i, 2])
            f.write(line)
        f.close()
    with open("ua.test", 'w') as f:
        for i in range(train.shape[0] - nb_test, train.shape[0]):
            line = "%d\t%d\t%d\n" % (train[i, 0], train[i, 1], train[i, 2])
            f.write(line)
        f.close()
    svd = SvdMatrix("aggregated_output_ids.base", 911, 1541)
    print("ok")
    svd.trainratings()
    print("rmsetrain: ", svd.calcrmse(svd.trainrats))
    svd.readtestsmaller("ua.test")
    print("rmsetest: ", svd.calcrmse(svd.testrats))
    print("time: ", time.time() - init)
    '''
    with open("aggregated_output_ids.base", 'w') as f:
        for i in range(train.shape[0]):
            line = "%d\t%d\t%d\n" % (train[i, 0], train[i, 1], train[i, 2])
            f.write(line)
        f.close()
    svd = SvdMatrix("aggregated_output_ids.base", 911, 1541, r=30)
    svd.trainratings()
    user_feature = np.matrix(svd.U)
    movie_feature = np.matrix(svd.V)
    sup_info = CollaborativeFiltering2.getSuppValues()
    with open("user_svd_features" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        users_offset = sup_info[0]
        line = "offset"
        for i in range(user_feature.shape[1]):
            line += ",feature_%d" % i
        line += "\n"
        f.write(line)
        for user in range(user_feature.shape[0]):
            line = "%f" % users_offset[user]
            for i in range(user_feature.shape[1]):
                line += ",%f" % user_feature[user, i]
            line += "\n"
            f.write(line)

    with open("movie_svd_features" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        movies_avg = sup_info[1]
        line = "avg"
        for i in range(movie_feature.shape[1]):
            line += ",feature_%d" % i
        line += "\n"
        f.write(line)
        for movie in range(movie_feature.shape[0]):
            line = "%f" % movies_avg[movie]
            for i in range(movie_feature.shape[1]):
                line += ",%f" % movie_feature[movie, i]
            line += "\n"
            f.write(line)

    with open("aggregated_svd_features_train" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        users_offset = sup_info[0]
        line = "offset"
        for i in range(user_feature.shape[1]):
            line += ",feature_%d" % i
        movies_avg = sup_info[1]
        line += ",avg"
        for i in range(movie_feature.shape[1]):
            line += ",feature_%d" % i
        line += "\n"
        f.write(line)
        for el in train:
            user = el[0] - 1
            movie = el[1] - 1
            line = "%f" % users_offset[user]
            for i in range(user_feature.shape[1]):
                line += ",%f" % user_feature[user, i]
            line += ",%f" % movies_avg[movie]
            for i in range(movie_feature.shape[1]):
                line += ",%f" % movie_feature[movie, i]
            line += "\n"
            f.write(line)
    test = pd.read_csv("data/data_test.csv", delimiter=",").values
    with open("aggregated_svd_features_test" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        users_offset = sup_info[0]
        line = "offset"
        for i in range(user_feature.shape[1]):
            line += ",feature_%d" % i
        movies_avg = sup_info[1]
        line += ",avg"
        for i in range(movie_feature.shape[1]):
            line += ",feature_%d" % i
        line += "\n"
        f.write(line)
        for el in test:
            user = el[0] - 1
            movie = el[1] - 1
            line = "%f" % users_offset[user]
            for i in range(user_feature.shape[1]):
                line += ",%f" % user_feature[user, i]
            line += ",%f" % movies_avg[movie]
            for i in range(movie_feature.shape[1]):
                line += ",%f" % movie_feature[movie, i]
            line += "\n"
            f.write(line)
