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

movies_genre = movies.loc[:,"Action":"Western"].values #gives for each film its genre
users_work = users.loc[:,"other":"none"].values #gives for each user its occupation
movies_avg_per_work = np.zeros((movies.shape[0],users_work.shape[1])) #gives for each film the avg is given by a work category
movies_view_per_work = np.zeros((movies.shape[0],users_work.shape[1])) #gives the number of times a film has been seen by a work category
users_avg_per_genre = np.zeros((users.shape[0], movies_genre.shape[1])) #gives for each user the avg note he gives to a film genre
users_view_per_genre = np.zeros((users.shape[0], movies_genre.shape[1])) #gives the number of times a user has watched a film of a certain genre
movies_genre_avg = np.zeros((movies_genre.shape[1],1)) #average of ratings per genre
movies_per_genre = np.zeros((movies_genre.shape[1],1))  #number of film watched in each genre
user_work_avg = np.zeros((users_work.shape[1],1)) #avg rating per work type
user_per_work = np.zeros((users_work.shape[1],1)) #number of users ratings per work type
rating_matrix = np.zeros((users.shape[0], movies.shape[0]))
k_movie = 15
k_user = 15

for el in train[:]:
    user_id = el[0]-1
    movie_id = el[1]-1
    rating = el[2]

    #get the user work
    i = 0
    for works in users_work[user_id]:
        if works == 1:
            movies_view_per_work[movie_id,i] += 1
            movies_avg_per_work[movie_id,i] += rating
        i += 1

    #get the film genre
    i = 0
    for genre in movies_genre[movie_id]:
        if genre == 1:
            users_view_per_genre[user_id,i] += 1
            users_avg_per_genre[user_id,i] += rating
        i += 1


movies_per_genre = users_view_per_genre.sum(axis=0) #!!! il y a certains films qui ont plusieurs genre...
movies_genre_avg = np.divide(users_avg_per_genre.sum(axis=0),movies_per_genre)
#division a la main pour eviter les divisions par 0
for i in range(users_avg_per_genre.shape[0]):
    for j in range(users_avg_per_genre.shape[1]):
        if(users_view_per_genre[i,j] != 0):
            users_avg_per_genre[i,j] /= users_view_per_genre[i,j]
print(users_avg_per_genre)
#users_avg_per_genre = np.divide((users_avg_per_genre + k_movie*movies_genre_avg), (users_view_per_genre+ k_movie))

user_per_work = movies_view_per_work.sum(axis=0)
user_work_avg = np.divide(movies_avg_per_work.sum(axis=0),user_per_work)
#division a la main pour eviter les division par 0
for i in range(movies_avg_per_work.shape[0]):
    for j in range(movies_avg_per_work.shape[1]):
        if(movies_view_per_work[i,j] != 0):
            movies_avg_per_work[i,j] /= movies_view_per_work[i,j]
print(movies_avg_per_work)

for user in range(users.shape[0]):
    for movie in range(movies.shape[0]):

        #get average according to the user work
        i = 0
        work_avg = 0
        nb_work = 0
        for works in users_work[user]:
            if works == 1:
                work_avg += movies_avg_per_work[movie, i]
                nb_work += 1
            i += 1
        if nb_work != 0:
            work_avg /= nb_work
        else:
            work_avg = np.mean(user_per_work);

        #get average for the film genre
        i = 0
        genre_avg = 0
        nb_genre = 0
        for genre in movies_genre[movie]:
            if genre == 1:
                genre_avg += users_avg_per_genre[user, i]
                nb_genre += 1
            i += 1
        #some films don't have a genre
        if nb_genre != 0:
            genre_avg /= nb_genre
        else:
            genre_avg = np.mean(movies_genre_avg)

        rating_matrix[user, movie] = (work_avg + genre_avg)/2

test = pd.read_csv("data/data_test.csv", delimiter=",").values
y = []
for el in test:
    y.append(rating_matrix[el[0] - 1, el[1] - 1])

make_submission(y, test[:, 0], test[:, 1])
