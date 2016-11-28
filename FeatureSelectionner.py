import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration

class FeatureSelectionner:

    def getImportanceFeatureByDecisionTree(self, feature_path):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')
        #print(features_train.as_matrix()[:5][:5])
        #print(features_train[:5][:5])
        clf = DecisionTreeClassifier(max_depth=20)
        x = self.featureAglom(feature_path, 20)
        #x = features_train.values;
        clf.fit(x, result_train.values)
        #clf.fit(features_train.values, result_train.values)
        for i in range(x.shape[1]):
            if clf.feature_importances_[i] > 0.06:
                print("score de la feature %d: %f" % (i, clf.feature_importances_[i]))

        #print("Score: %f" % clf.score(features_train.values, result_train.values))
        print("Score: %f" % clf.score(x, result_train.values))
        nb_fold = 10
        print(features_train.values.shape)
        cl = DecisionTreeClassifier(max_depth=20)
        #score = sum(cross_val_score(clf, features_train.values, result_train.values, cv=nb_fold)) / nb_fold
        score = sum(cross_val_score(cl, x, result_train["rating"], cv=nb_fold)) / nb_fold
        print("Cross validation score: %f" % score)

    def featureAglom(self, feature_path, n_feature):
        features_train = pd.read_csv(feature_path, delimiter=',')
        result_train = pd.read_csv('data/output_train.csv', delimiter=',')
        clf = FeatureAgglomeration(n_clusters=n_feature)
        new_feature = clf.fit_transform(features_train.values, result_train.values)
        print(new_feature.shape)
        return new_feature

test = FeatureSelectionner()
test.getImportanceFeatureByDecisionTree("agregated_data_28-11-2016_01h50.csv")
#test.featureAglom("agregated_data_28-11-2016_01h50.csv")