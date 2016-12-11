#See http://machinelearningmastery.com/feature-selection-machine-learning-python/ for more

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

def printSelFeatNScore(ranking, n, feat_names):
    feat_n_score = np.zeros((n,2))
    if n < ranking.__len__:
        ind = np.argpartition(ranking, -n)[-n:]
        ind = np.sort(ind, 0)
        feat_n_score[:,0] = ind
        feat_n_score[:,1] = ranking[ind]
        sel_feat_names = feat_names[ind]
        for el in sel_feat_names:
            print('"%s",' % el)
        print(feat_n_score)


#load data
output = pd.read_csv("data/output_train.csv", delimiter=",")["rating"]
aggregated = pd.read_csv("data/agregated_data_28-11-2016_01h50.csv", delimiter=",")
names = aggregated.keys()
#output = preprocessing.scale(output.values)
#aggregated = preprocessing.scale(aggregated.values)
nb_feat = 6
test = 5
all = 1

if all == 1 or test==1:
    # Filter method
    # Feature Extraction with Univariate Statistical Tests
    print("UST")
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    #Can use a lot of different class like SelectKBest, SelectPercentil, SelectFpr, SelectFdr, ...
    test = SelectKBest(score_func=f_regression, k=nb_feat) #Score_func has to change if you want to do clasification or regression
    fit = test.fit(aggregated, output)
    features = fit.transform(aggregated.values)
    printSelFeatNScore(fit.scores_, nb_feat, names)

if all == 1 or test == 2:
    # Wrapper method
    # Feature Extraction with RFE
    print("\nRFE")
    from sklearn.feature_selection import RFE
    model = DecisionTreeClassifier(max_depth=20) #can use different algorithm but not all
    rfe = RFE(model, nb_feat)
    fit = rfe.fit(aggregated, output)
    features = fit.transform(aggregated.values)
    printSelFeatNScore(fit.ranking_, nb_feat, names)

if all == 1 or test == 3:
    # Aggregation of features
    # Feature Extraction with Principal Component Analysis
    print("\nPCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=nb_feat) #pas simple de comprendre tous les parametres
    fit = pca.fit(aggregated) #resultats pas fort intepretable mais pas mal d'autres fonctions qui pourraient etre utiles
    print("Explained Variance: %s") % fit.explained_variance_ratio_

if all == 1 or test == 4:
    # Bagged Decision Trees
    # Feature Importance with Extra Trees Classifier
    print("\nET")
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier() #loads of parameters
    model.fit(aggregated, output)
    printSelFeatNScore(model.feature_importances_, nb_feat, names)

if all == 1 or test == 5:
    # Bagged Decision Trees
    # Feature Importance with Random forest
    print("\nRF")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor() #loads of parameters
    model.fit(aggregated, output)
    printSelFeatNScore(model.feature_importances_, nb_feat, names)
