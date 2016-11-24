#! /usr/bin/env python
# -*- coding: utf-8 -*-


## Import packages
import os
import numpy as np
import pandas as pd

import time
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
            line = '{:0.0f}_{:0.0f},{}\n'.format(user_id_test[i],movie_id_test[i],y_predict[i])
            f.write(line)
    print("Submission file successfully written!")


if __name__ == "__main__":


    # Load data_train
    data_train = pd.read_csv('data/data_train.csv', delimiter=',')

    # Load output_train
    output_train = pd.read_csv('data/output_train.csv', delimiter=',')

    # Load data_test
    data_test = pd.read_csv('data/data_test.csv', delimiter=',')

    # Load user info
    data_user = pd.read_csv('data/data_user.csv', delimiter=',')

    # Load movie info
    data_movie = pd.read_csv('data/data_movie.csv', delimiter=',')

    # Create matrix "users x movies" for training data

    n_users = len(np.unique(data_user['user_id']))
    n_movies = len(np.unique(data_movie['movie_id']))

    matrix = np.zeros((n_users, n_movies))

    print("Building the matrix...")
    for i in range(data_train.shape[0]):
        user_idx = data_train.loc[i]['user_id']-1
        item_idx = data_train.loc[i]['movie_id']-1
        rating_idx = output_train.loc[i]['rating']
        matrix[int(user_idx), int(item_idx)] = int(rating_idx)    

    matrix[matrix==0] = np.nan

    # Toy prediction
    n_elements = data_test.shape[0]
    
    y_predict = np.ones((n_elements,)) * 5

    # Make a submission file

    make_submission(y_predict, name="toy_submission", user_id_test=data_test['user_id'], movie_id_test=data_test['movie_id'])
