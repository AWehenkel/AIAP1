import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
import time
import numpy as np

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
        #print(features_train.as_matrix()[:5][:5])
        #print(features_train[:5][:5])
        clf = DecisionTreeClassifier(max_depth=20)
        aglomerator = self.featureAglom(feature_path, 10)
        x = aglomerator.transform(features_train.values)
        #x = features_train.values;
        clf.fit(x, result_train.values)
        for i in range(x.shape[1]):
            if clf.feature_importances_[i] > 0.06:
                print("score de la feature %s: %f" % (features_train.keys()[i], clf.feature_importances_[i]))

        #print("Score: %f" % clf.score(features_train.values, result_train.values))
        print("Score: %f" % clf.score(x, result_train.values))
        nb_fold = 10
        print(features_train.values.shape)
        cl = DecisionTreeClassifier(max_depth=20)
        #score = sum(cross_val_score(clf, features_train.values, result_train.values, cv=nb_fold)) / nb_fold
        score = sum(cross_val_score(cl, x, result_train["rating"], cv=nb_fold)) / nb_fold
        print("Cross validation score: %f" % score)
        return (clf, aglomerator)

    def featureAglom(self, feature_path, n_feature):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')
        clf = FeatureAgglomeration(n_clusters=n_feature)
        clf.fit(features_train.values, result_train.values)
        return clf

test = FeatureSelectionner()
algos = test.getImportanceFeatureByDecisionTree("agregated_data_28-11-2016_01h50.csv")
test_set = pd.read_csv('agregated_data_test_28-11-2016_11h11.csv', delimiter=',')
test_set_id = pd.read_csv('data/data_test.csv', delimiter=',')
result = algos[0].predict(algos[1].transform(test_set.values))
make_submission(result, test_set_id["user_id"], test_set_id["movie_id"], "perfect")
#test.featureAglom("agregated_data_28-11-2016_01h50.csv")