from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as knn
import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
class CollaborativeFiltering:
    def __init__(self, user_data, movie_data, train_data, output_train, n_user_neighbor = 5, n_movie_neighbor = 5, nb_info_user = 10, nb_info_film = 10):
        self.train = train_data
        self.users = user_data
        self.movies = movie_data
        self.output_train = output_train

        self.nb_info_user = nb_info_user
        self.nb_info_film = nb_info_film

        self.n_user_neigbors = n_user_neighbor
        self.n_movie_neighbor = n_movie_neighbor

        self.knn_user = NearestNeighbors(n_neighbors=n_user_neighbor, algorithm='ball_tree')
        self.knn_user.fit(user_data)


    def getUserN(self, user):
        return self.knn_user.kneighbors(user)

    def score(self, test_data, y):
        result = self.predict(test_data)
        return ((y - result)**2).mean()

    def getUserMovies(self, user):
        cond = self.train[:, 0] == (user + 1)
        movies_id = np.extract(cond, self.train[:, 1])
        ratings = np.extract(cond, self.output_train)
        return (movies_id, ratings)

    def predict(self, test_data):
        cur_user = -1
        prediction = np.zeros((test_data.shape[0]))
        i = 0
        for couple in test_data:
            user = self.users[couple[0] - 1]
            if(couple[0] - 1 != cur_user):
                movie = self.movies[couple[1] - 1]
                same_users = self.knn_user.kneighbors([user])[1]
                movies_user = []
                movies_rating = []
                for n in same_users[0]:
                    results = self.getUserMovies(n)
                    movies_user = np.append(movies_user, results[0])
                    movies_rating = np.append(movies_rating, results[1])
                nn = knn.KNeighborsRegressor(n_neighbors=self.n_movie_neighbor)
                nn.fit(self.movies[movies_user.astype(int) - 1],movies_rating)
            prediction[i] = nn.predict([movie])
            if(i % 1000 == 0 and i > 10):
                print(i)
            i = i + 1
        return prediction


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

users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
train = pd.read_csv("data/data_train.csv", delimiter=",")
output = pd.read_csv("data/output_train.csv", delimiter=",")
nb_estim = 10
for nn in range(5, 20):
    score = 0.0
    for i in range(nb_estim):
        rand = random.sample(range(1, train.values.shape[0]), int(train.values.shape[0]/2))
        model = CollaborativeFiltering(users.values[:, 1:], movies.values[:, 1:], train.values[rand], output.values[rand], n_movie_neighbor=15, n_user_neighbor=15)
        #print(model.getUserMovies(0))
        test = pd.read_csv("data/data_test.csv")
        data = test.values
        #result = model.predict(data)
        rand = random.sample(range(1, train.values.shape[0]), 1000)
        score = score + model.score(train.values[rand], output.values[rand])/nb_estim
    print("Score pour %d neigbhbors: %f" % (nn, score))


#make_submission(result, data[:, 0], data[:, 1], "data/results/collaborative")

