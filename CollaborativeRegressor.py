import sklearn.neighbors as knn
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


class CollaborativeRegressor:
    def __init__(self, movie_data, user_data, nb_neighbors_user, nb_neighbors_movie):
        self.movies = movie_data
        self.users = user_data

        self.nb_users = nb_neighbors_user
        self.nb_movies = nb_neighbors_movie

        self.knn_user = knn.NearestNeighbors(n_neighbors=nb_neighbors_user, algorithm='brute', metric='cosine')
        self.knn_user.fit(user_data)


    def fit(self, x, y):
        if(x.shape[1] != 2):
            raise AttributeError("x should contains two features")
        self.x = x
        self.ratings = y
        self.user_movies = {}
        i = 0
        tmp = 0
        for el in x:
            tmp = el[0] - 1
            if tmp in self.user_movies:
                self.user_movies[tmp].append([el[1], y[i]])
            else:
                self.user_movies[tmp] = [[el[1], y[i]]]
            i = i + 1

    def predict(self, x):
        old_user = -1
        y = []
        if (self.x.shape[1] != x.shape[1]):
            raise AttributeError("Fit format different from predict format")
        for el in x:
            movie = self.movies[el[1] - 1]
            user = self.users[el[0] - 1]
            user_id = el[0] - 1
            if(user_id != old_user):
                old_user = user_id
                #Il faut s'assurer qu'on a le nombre de neighbors voulu.
                nb_neigbors = 0
                i = 0
                j = 0
                user_neighbors = []
                while (nb_neigbors < self.nb_users):
                    tmp_neighbors = self.knn_user.kneighbors([user], return_distance=False, n_neighbors=(self.nb_users + i))[0]
                    for n in tmp_neighbors[(self.nb_users - 1) * j + i:]:
                        if n in self.user_movies:
                            nb_neigbors = nb_neigbors + 1
                            user_neighbors.append(n)
                    i = i + 1
                    j = 1
            movies_user = []
            movies_rating = []
            for u in user_neighbors:
                results = self.user_movies[u]
                results = np.array(results)
                movies_user = np.append(movies_user, results[:, 0])
                movies_rating = np.append(movies_rating, results[:, 1])
            nn = knn.KNeighborsRegressor(n_neighbors=min(self.nb_movies, len(movies_rating)), algorithm='brute', metric='cosine')
            nn.fit(self.movies[movies_user.astype(int) - 1], movies_rating)
            y.append(nn.predict([movie]))

        return y

    def score(self, x, y):
        result = self.predict(x)
        print("ok")
        return ((y - result) ** 2).mean()

    def get_params(self, deep = False):
        return {'movie_data': self.movies, 'user_data' : self.users, 'nb_neighbors_user' : self.nb_users, 'nb_neighbors_movie' : self.nb_movies}


users = pd.read_csv("user_svd_features_11-12-2016_17h36.csv", delimiter=",")
users = users.values[:, 1:]
movies = pd.read_csv("movie_svd_features_11-12-2016_17h36.csv", delimiter=",")
movies = movies.values[:, 1:]
train = pd.read_csv("data/data_train.csv", delimiter=",")
output = pd.read_csv("data/output_train.csv", delimiter=",")
t_o = np.append(train, output, axis=1)
np.random.shuffle(t_o)
t = CollaborativeRegressor(preprocessing.scale(movies), preprocessing.scale(users), 20, 40)
score = cross_val_score(t, t_o[:, :2], t_o[:, 2:], cv=10).mean()
print("Validation score %f" % score)
best_score = 10
best_user_n = 0
best_movie_n = 0
for u in range(1, 10):
    for m in range(1, 10):
        t = CollaborativeRegressor(preprocessing.scale(movies), preprocessing.scale(users), u*10, m*10)
        t.fit(t_o[5000:, :2], t_o[5000:, 2:])
        score =  t.score(t_o[:5000, :2], t_o[:5000, 2:])
        if score < best_score:
            best_score = score
            best_user_n = u
            best_movie_n = m
            print("Best score for %d %d (%f)" % (best_user_n, best_movie_n, best_score))
print("Best score for %d %d (%f)" % (best_user_n, best_movie_n, best_score))
#print(train.values[1000:1010, :])
#t.predict(train.values[1000:1010, :])
'''
best_score = 10
best_user_n = 0
best_movie_n = 0
for i in range(15, 16):
    for j in range(15, 16):
        t = CollaborativeRegressor(movies, users, i, j)
        score = cross_val_score(t, train.values[:], output.values[:, 0], cv = 10).mean()
        if score < best_score:
            best_score = score
            best_user_n = i
            best_movie_n = j

print("Best score for %d %d (%f)" % (best_user_n, best_movie_n, best_score))
'''
t = CollaborativeRegressor(preprocessing.scale(movies), preprocessing.scale(users), best_user_n*10, best_movie_n*10)
score = cross_val_score(t, t_o[:, :2], t_o[:, 2:], cv=10).mean()
print("Validation score %f" % score)