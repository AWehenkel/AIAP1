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
        algos = test.getImportanceFeatureByDecisionTree(feature_path)
        print("random forest")
        bag1 = ens.RandomForestRegressor(n_estimators=15)
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

    def adaboosTree(self, feature_path):
        x = pd.read_csv(feature_path, delimiter=',').values
        y = pd.read_csv('data/output_train.csv', delimiter=',')["rating"].values
        print("Feature aglom:")
        aglomerator = self.featureAglom(feature_path, 25)
        x_aglom = aglomerator.transform(x)
        print("forest construction")
        clf = ens.AdaBoostRegressor(n_estimators=10, base_estimator=DecisionTreeRegressor(max_depth=10), loss='exponential')
        clf.fit(x_aglom,y)
        print(clf.score(x_aglom,y))
        nb_fold = 10
        #scores = cross_val_score(clf, x_aglom, y, scoring="neg_mean_squared_error", cv=nb_fold)
        #print(scores)
        #print(scores.mean())
        return (clf, aglomerator)

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
        algos = test.getImportanceFeatureByDecisionTree(feature_path)
        mlp = nn.MLPRegressor(hidden_layer_sizes=3)
        mlp.fit(algos[2], algos[3])
        print("Score mlp: %f" % mlp.score(algos[2], algos[3]))
        for n in range(1, 5):
            cl = nn.MLPRegressor(hidden_layer_sizes=n)
            nb_fold = 10
            score = sum(cross_val_score(cl, algos[2], algos[3], cv=nb_fold, scoring='neg_mean_squared_error')) / nb_fold
            print("%d: Cross validation score: %f" % (n, score))
        return (mlp, algos[1])

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

    def featureAglom(self, feature_path, n_feature):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')
        clf = FeatureAgglomeration(n_clusters=n_feature)
        #clf = deco.PCA()
        clf.fit(features_train.values, result_train.values)
        return clf

test = FeatureSelectionner()
#algos = test.getImportanceFeatureByDecisionTree("agregated_data_28-11-2016_01h50.csv")
#algos = test.makeKNNRegression("agregated_data_28-11-2016_01h50.csv")
#algos = test.baggingKNNRegression("agregated_data_28-11-2016_01h50.csv")
#algos = test.randomForest("agregated_data_28-11-2016_01h50.csv")
algos = test.adaboosTree("agregated_data_28-11-2016_01h50.csv")
test_set = pd.read_csv('agregated_data_test_28-11-2016_11h11.csv', delimiter=',')
test_set_id = pd.read_csv('data/data_test.csv', delimiter=',')
print("Compute prediciton...")
result = algos[0].predict(algos[1].transform(test_set.values))
#result = algos[0].predict(algos[1].transform(test_set.values))
make_submission(result, test_set_id["user_id"], test_set_id["movie_id"], "perfect")
#test.featureAglom("agregated_data_28-11-2016_01h50.csv")