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

train = pd.read_csv("data/data_train.csv", delimiter=",")
train_values = pd.read_csv("data/output_train.csv", delimiter=",")
train = np.append(train, train_values, axis=1)
users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
print(users.shape[0])
users_offset = np.zeros(users.shape[0]) #for each user, the offset from the average ranking of the movies
users_nb_movie = np.zeros(users.shape[0]) #for each user, number of movies that he watched
movies_avg = np.zeros(movies.shape[0]) #for each movie, the average of the ranking
movies_nb_user = np.zeros(movies.shape[0]) #for each movie, number of users that watched it
rating_matrix = np.zeros((users.shape[0], movies.shape[0]))
#Creation des matrice offset et mean
for el in train[:]:
    users_nb_movie[el[0] - 1] = users_nb_movie[el[0] - 1] + 1
    users_offset[el[0] - 1] = users_offset[el[0] - 1] + el[2]
    movies_nb_user[el[1] - 1] = movies_nb_user[el[1] - 1] + 1
    movies_avg[el[1] - 1] = movies_avg[el[1] - 1] + el[2]

# k is proportional the inverse of the wieght given to one movie/user
# if k = 0, you only focus on the data for that movie/user
# if k = inf, you only focus on the average of all movie/users
k_movie = 15
global_avg_movie = movies_avg.sum()/movies_nb_user.sum() #the average mark for all movies
print(global_avg_movie)
k_user = 15
global_avg_user = users_offset.sum()/users_nb_movie.sum() # same as global_avg_movie, logical
movies_avg = np.divide((movies_avg + k_movie*global_avg_movie), (movies_nb_user + k_movie))
print(movies_avg)
users_offset = np.divide((users_offset + k_user*global_avg_user), (k_user + users_nb_movie)) - movies_avg.mean() #pourquoi on utilises ca?
for user in range(users.shape[0]):
    for movie in range(movies.shape[0]):
        rating_matrix[user, movie] = movies_avg[movie] + users_offset[user]


test = pd.read_csv("data/data_test.csv", delimiter=",").values
y = []
for el in test:
    y.append(rating_matrix[el[0] - 1, el[1] - 1])


#make_submission(y, test[:, 0], test[:, 1])