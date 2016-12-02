import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
import time
import numpy as np
import sklearn.neighbors as knn
import sklearn.neural_network as nn
import sklearn.ensemble as ens
import sklearn.decomposition as deco
import sklearn.svm as svm

def make_submission(y_predict, user_id_test, movie_id_test, name=None, date=True):
    n_elements = len(y_predict)

    if name is None:
      name = 'submission'
    if date:
      name = name + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M'))

    with open(name + ".csv", 'w') as f:
        f.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for i in range(n_elements):
            if np.isnan(y_predict[i]):
                raise ValueError('NaN detected!')
            line = '{:0.0f}_{:0.0f},{:0.1f}\n'.format(user_id_test[i],movie_id_test[i],y_predict[i])
            f.write(line)
    print("Submission file successfully written!")

class FeatureSelectionner:

    def getImportanceFeatureByDecisionTree(self, feature_path):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')
        clf = DecisionTreeClassifier(max_depth=20)
        aglomerator = self.featureAglom(feature_path, 20)
        x = aglomerator.transform(features_train.values)
        clf.fit(x, result_train.values)
        y = result_train["rating"].values
        important_feature = np.zeros(x.shape)
        h = 0
        for i in range(x.shape[1]):
            if clf.feature_importances_[i] > 0.06:
                print("score de la feature %s: %f" % (features_train.keys()[i], clf.feature_importances_[i]))
                important_feature[:][h] = x[:][h]
                h += 1
        print("Score: %f" % clf.score(x, result_train.values))
        nb_fold = 10
        #cl = DecisionTreeClassifier(max_depth=20)
        #score = sum(cross_val_score(cl, x, result_train["rating"], cv=nb_fold)) / nb_fold
        #print("Cross validation score: %f" % score)
        return (clf, aglomerator, x, y)

    def randomForest(self, feature_path):
        #algos = test.getImportanceFeatureByDecisionTree(feature_path)
        x = pd.read_csv(feature_path, delimiter=',')
        result = pd.read_csv('data/output_train.csv', delimiter=',')
        y = result["rating"].values
        print("random forest")
        bag1 = ens.RandomForestRegressor(n_estimators=5)
        bag1.fit(x, y)
        print("Random forest ok")
        importances = bag1.feature_importances_
        indices = np.argsort(importances)[::-1]
        keys = pd.read_csv(feature_path, delimiter=',').keys()
        for f in range(x.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, keys[indices[f]], importances[indices[f]]))
        nb_fold = 10
        score = sum(cross_val_score(bag1, x, y, cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
        print("Cross validation score: %f" % score)
        #nb_fold = 10
        #score = sum(cross_val_score(bag1, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
        #print("Cross validation score on bagging: %f" % score)
        '''
        for n in range(5, 20):
            cl = knn.KNeighborsRegressor(n_neighbors=n)
            bag = ens.BaggingRegressor(cl)
            nb_fold = 10
            score = sum(cross_val_score(bag, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
            print("%d: Cross validation score: %f" % (n, score))
        '''
        #return (bag1, algos[1])

    def SVR(self, feature_path):
        x = pd.read_csv(feature_path, delimiter=',').values
        y = pd.read_csv('data/output_train.csv', delimiter=',')["rating"].values
        print("Feature aglom:")
        aglomerator = self.featureAglom(feature_path, 7)
        x_aglom = aglomerator.transform(x)
        print("SVR construction")
        clf = svm.SVR()
        clf.fit(x_aglom, y)
        print("SVR constructed")
        nb_fold = 2
        scores = cross_val_score(clf, x_aglom, y, scoring="neg_mean_squared_error", cv=nb_fold)
        print(scores.mean())

    def adaboosTree(self, feature_path):
        x = pd.read_csv(feature_path, delimiter=',').values
        y = pd.read_csv('data/output_train.csv', delimiter=',')["rating"].values
        print("Feature aglom:")
        aglomerator = self.featureAglom(feature_path, 10)
        x_aglom = aglomerator.transform(x)
        print("forest construction")
        clf = ens.AdaBoostRegressor(n_estimators=1000, base_estimator=DecisionTreeRegressor(max_depth=3), loss='exponential')
        clf.fit(x_aglom,y)
        print(clf.score(x_aglom,y))
        nb_fold = 10
        scores = cross_val_score(clf, x_aglom, y, scoring="neg_mean_squared_error", cv=nb_fold)
        print(scores.mean())
        return (clf, x)

    def makeKNNRegression(self, feature_path):
        algos = test.getImportanceFeatureByDecisionTree(feature_path)
        kn = knn.KNeighborsRegressor(n_neighbors=5)
        kn.fit(algos[2], algos[3])
        print("Score knn: %f" % kn.score(algos[2], algos[3]))
        for n in range(5,20):
            cl = knn.KNeighborsRegressor(n_neighbors=n)
            nb_fold = 10
            score = sum(cross_val_score(cl, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
            print("%d: Cross validation score: %f" % (n, score))
        return (kn, algos[1])

    def makeMLP(self, feature_path):
        x = pd.read_csv(feature_path, delimiter=',').values
        y = pd.read_csv('data/output_train.csv', delimiter=',')["rating"].values
        print("MLP aglom:")
        #mlp = nn.MLPRegressor(hidden_layer_sizes=1)
        #mlp.fit(x, y)
        #print("Score mlp: %f" % mlp.score(x, y))
        for n in range(1, 15):
            cl = nn.MLPRegressor(hidden_layer_sizes=n)
            bag = ens.AdaBoostRegressor(cl, 5, loss="exponential")
            cl = bag
            nb_fold = 10
            score = sum(cross_val_score(cl, x, y, cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
            print("%d: Cross validation score: %f" % (n, score))
        #return (mlp, x)

    def baggingKNNRegression(self, feature_path):
        algos = test.getImportanceFeatureByDecisionTree(feature_path)
        kn = knn.KNeighborsRegressor(n_neighbors=25)
        bag1 = ens.BaggingRegressor(kn)
        bag1.fit(algos[2], algos[3])
        nb_fold = 10
        score = sum(cross_val_score(bag1, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
        print("Cross validation score on bagging: %f" % score)
        '''
        for n in range(5, 20):
            cl = knn.KNeighborsRegressor(n_neighbors=n)
            bag = ens.BaggingRegressor(cl)
            nb_fold = 10
            score = sum(cross_val_score(bag, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
            print("%d: Cross validation score: %f" % (n, score))
        '''
        return (bag1, algos[1])

    def featureAglom(self, feature_path, n_feature = 10):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')["rating"]
        clf = FeatureAgglomeration(n_clusters=n_feature)
        #clf = deco.PCA()
        clf.fit(features_train.values, result_train.values)
        return clf

test = FeatureSelectionner()
#algos = test.getImportanceFeatureByDecisionTree("agregated_data_28-11-2016_01h50.csv")
#algos = test.makeKNNRegression("agregated_data_28-11-2016_01h50.csv")
#algos = test.baggingKNNRegression("agregated_data_28-11-2016_01h50.csv")
#algos = test.randomForest("agregated_data_28-11-2016_01h50.csv")
#algos = test.makeMLP("agregated_data_28-11-2016_01h50.csv")
#algos = test.adaboosTree("agregated_data_28-11-2016_01h50.csv")
#algos = test.SVR("agregated_data_28-11-2016_01h50.csv")
#test_set = pd.read_csv('agregated_data_test_28-11-2016_11h11.csv', delimiter=',')
#test_set_id = pd.read_csv('data/data_test.csv', delimiter=',')
#print("Compute prediciton...")
#result = algos[0].predict(algos[1].transform(test_set.values))
#result = algos[0].predict(algos[1].transform(test_set.values))
#make_submission(result, test_set_id["user_id"], test_set_id["movie_id"], "perfect")
#test.featureAglom("agregated_data_28-11-2016_01h50.csv")
x = pd.read_csv("agregated_data_28-11-2016_01h50.csv", delimiter=',')
'''
for i in range(x.shape[0]):
    print(x.keys()[i])
    if(i == 0):
        print(max(x.values[i][1:]))
    elif(i==x.shape[0]-1):
        print(max(x.values[i][:-1]))
    else:
        print(max((max(x.values[i][i+1:]), max(x.values[i][:i]))))
'''
t = 20
clf = test.featureAglom("agregated_data_28-11-2016_01h50.csv", n_feature=t)
clusters = clf.labels_
for i in range(t):
    print("Cluster %d:" % i)
    for j in range(clf.labels_.shape[0]):
        if clf.labels_[j] == i:
            print("\t%s" % x.keys()[j])
