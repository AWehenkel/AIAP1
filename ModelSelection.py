import numpy as np
import pandas as pd
import datetime
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import KFold

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
            line = '{:0.0f},{:0.0f},{}\n'.format(user_id_test[i],movie_id_test[i],y_predict[i])
            f.write(line)
    print("Submission file successfully written!")

class ModelSelection:
    def __init__(self, user_data, movie_data, aggregated_data, train_data, output_train):
        self.train = train_data
        self.users = user_data
        self.aggregated = aggregated_data
        self.movies = movie_data
        self.output_train = output_train

    def optimizeParametersDecisionTreeClassifier(self, nb_fold, max_depth_range):

        kf = KFold(n_splits=nb_fold)
        depth_n_errors = np.zeros((max_depth_range.__len__(), 2))
        i = 0
        for depth in max_depth_range:
            depth_n_errors[i][0] = depth
            i += 1
        #First round of cv
        for train_index, test_index in kf.split(self.aggregated):
            #Second round of cv
            i = 0
            for depth in max_depth_range:
                dt = DecisionTreeClassifier(max_depth=depth)
                scores = cross_val_score(dt, self.aggregated[train_index], self.output_train[train_index], cv=nb_fold, scoring='neg_mean_squared_error')
                depth_n_errors[i][1] += -scores.mean()
                i += 1

        i = 0
        for depth in max_depth_range:
            depth_n_errors[i][1] /= nb_fold
            i += 1

        best_depth = 0
        best_error = 5
        #Take the best model and cross validate it on the whole data
        for depth, error in depth_n_errors:
            if(error < best_error):
                best_error = error
                best_depth = depth

        #Recompute the error for this model on the whole data set
        dt = DecisionTreeClassifier(max_depth=best_depth)
        final_error = -cross_val_score(dt, self.aggregated, self.output_train, cv=nb_fold, scoring='neg_mean_squared_error')

        return[best_depth, final_error.mean()]


    def optimizeParametersKNeighborsClassifier(self, nb_fold, k_range):

        kf = KFold(n_splits=nb_fold)
        k_n_errors = np.zeros((k_range.__len__(), 2))
        i = 0
        for k in k_range:
            k_n_errors[i][0] = k
            i += 1
        #First round of cv
        for train_index, test_index in kf.split(self.aggregated):
            #Second round of cv
            i = 0
            for k in k_range:
                dt = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(dt, self.aggregated[train_index], self.output_train[train_index], cv=nb_fold, scoring='neg_mean_squared_error')
                k_n_errors[i][1] += -scores.mean()
                i += 1

        for i in range(k_range.__len__()):
            k_n_errors[i][1] /= nb_fold

        best_k = 0
        best_error = 5
        #Take the best model and cross validate it on the whole data
        for k, error in k_n_errors:
            if(error < best_error):
                best_error = error
                best_k = k

        #Recompute the error for this model on the whole data set
        dt = KNeighborsClassifier(n_neighbors=best_k)
        final_error = -cross_val_score(dt, self.aggregated, self.output_train, cv=nb_fold, scoring='neg_mean_squared_error')

        return[best_k, final_error.mean()]

    def optimizeParametersKNeighborsRegressor(self, nb_fold, k_range):

        kf = KFold(n_splits=nb_fold)
        k_n_errors = np.zeros((k_range.__len__(), 2))
        i = 0
        for k in k_range:
            k_n_errors[i][0] = k
            i += 1
        #First round of cv
        for train_index, test_index in kf.split(self.aggregated):
            #Second round of cv
            i = 0
            for k in k_range:
                dt = KNeighborsRegressor(n_neighbors=k)
                scores = cross_val_score(dt, self.aggregated[train_index], self.output_train[train_index], cv=nb_fold, scoring='neg_mean_squared_error')
                k_n_errors[i][1] += -scores.mean()
                i += 1

        for i in range(k_range.__len__()):
            k_n_errors[i][1] /= nb_fold

        best_k = 0
        best_error = 5
        #Take the best model and cross validate it on the whole data
        for k, error in k_n_errors:
            if(error < best_error):
                best_error = error
                best_k = k

        #Recompute the error for this model on the whole data set
        dt = KNeighborsRegressor(n_neighbors=best_k)
        final_error = -cross_val_score(dt, self.aggregated, self.output_train, cv=nb_fold, scoring='neg_mean_squared_error')

        return[best_k, final_error.mean()]




users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
train = pd.read_csv("data/data_train.csv", delimiter=",")
output = pd.read_csv("data/output_train.csv", delimiter=",")["rating"]
aggregated = pd.read_csv("data/agregated_data_28-11-2016_01h50.csv", delimiter=",")
ms = ModelSelection(users.values, movies.values, aggregated.values, train.values, output)
#print(ms.optimizeParametersDecisionTreeClassifier(5, range(2,3,1)))
print(ms.optimizeParametersKNeighborsClassifier(5, range(1,5,1)))
#print(ms.optimizeParametersKNeighborsClassifier(5, range(5,10,1)))