import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
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
def getSuppValues():
    train = pd.read_csv("data/data_train.csv", delimiter=",")
    train_values = pd.read_csv("data/output_train.csv", delimiter=",")
    train = np.append(train, train_values, axis=1)
    users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
    movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
    print(users.shape[0])
    users_offset = np.zeros(users.shape[0])
    users_nb_movie = np.zeros(users.shape[0])
    movies_avg = np.zeros(movies.shape[0])
    movies_nb_user = np.zeros(movies.shape[0])
    rating_matrix = np.zeros((users.shape[0], movies.shape[0]))
    #Création des matrice offset et mean
    for el in train[:]:
        users_nb_movie[el[0] - 1] = users_nb_movie[el[0] - 1] + 1
        users_offset[el[0] - 1] = users_offset[el[0] - 1] + el[2]
        movies_nb_user[el[1] - 1] = movies_nb_user[el[1] - 1] + 1
        movies_avg[el[1] - 1] = movies_avg[el[1] - 1] + el[2]

    k_movie = 20
    global_avg_movie = movies_avg.sum()/movies_nb_user.sum()

    k_user = 20
    global_avg_user = users_offset.sum()/users_nb_movie.sum()
    movies_avg = np.divide((movies_avg + k_movie*global_avg_movie), (movies_nb_user + k_movie))

    users_offset = np.divide((users_offset + k_user*global_avg_user), (k_user + users_nb_movie)) - movies_avg.mean()
    return(users_offset, movies_avg)

def getAvgRating():
    train = pd.read_csv("data/data_train.csv", delimiter=",")
    train_values = pd.read_csv("data/output_train.csv", delimiter=",")
    train = np.append(train, train_values, axis=1)
    users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
    movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
    ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Film-Noir',
     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Documentary', 'Drama',
     'Fantasy']



test = pd.read_csv("data/data_test.csv", delimiter=",").values
y = []
#for el in test:
#    y.append(rating_matrix[el[0] - 1, el[1] - 1])


#make_submission(y, test[:, 0], test[:, 1])




