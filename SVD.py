import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import time
import CollaborativeFiltering2

train = pd.read_csv("data/data_train.csv", delimiter=",")
train_values = pd.read_csv("data/output_train.csv", delimiter=",")
train = np.append(train, train_values, axis=1)
users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
test = pd.read_csv("data/data_test.csv", delimiter=",").values

user_feature = np.zeros((users.shape[0], 20)) + 0.1
movie_feature = np.zeros((movies.shape[0], 20)) + 0.1

nb_loop = 10
lrate = 0.01
k = 0.02
for loop in range(nb_loop):
    avg_error = 0
    for el in train:
        user = el[0] - 1
        movie = el[1] - 1
        rating = el[2]
        predicted_rating = np.multiply(user_feature[user, :], movie_feature[movie, :])
        predicted_rating = np.sum(predicted_rating)
        error = lrate * (rating - predicted_rating)
        avg_error += (rating - predicted_rating)

        uv = user_feature[user, :]
        user_feature[user, :] += error*movie_feature[movie, :]
        movie_feature[movie, :] += error*uv
    print(avg_error/len(train_values))

sup_info = CollaborativeFiltering2.getSuppValues()
with open("user_svd_features" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
    users_offset = sup_info[0]
    line = "offset"
    for i in range(user_feature.shape[1]):
        line += ",feature_%d" % i
    line += "\n"
    f.write(line)
    for user in range(user_feature.shape[0]):
        line = "%f" % users_offset[user]
        for i in range(user_feature.shape[1]):
            line += ",%f" % user_feature[user, i]
        line += "\n"
        f.write(line)

with open("movie_svd_features" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
    movies_avg = sup_info[1]
    line = "avg"
    for i in range(movie_feature.shape[1]):
        line += ",feature_%d" % i
    line += "\n"
    f.write(line)
    for movie in range(movie_feature.shape[0]):
        line = "%f" % movies_avg[movie]
        for i in range(movie_feature.shape[1]):
            line += ",%f" % movie_feature[movie, i]
        line += "\n"
        f.write(line)

with open("aggregated_svd_features_train" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
    users_offset = sup_info[0]
    line = "offset"
    for i in range(user_feature.shape[1]):
        line += ",feature_%d" % i
    movies_avg = sup_info[1]
    line += ",avg"
    for i in range(movie_feature.shape[1]):
        line += ",feature_%d" % i
    line += "\n"
    f.write(line)
    for el in train:
        user = el[0] - 1
        movie = el[1] - 1
        line = "%f" % users_offset[user]
        for i in range(user_feature.shape[1]):
            line += ",%f" % user_feature[user, i]
        line += ",%f" % movies_avg[movie]
        for i in range(movie_feature.shape[1]):
            line += ",%f" % movie_feature[movie, i]
        line += "\n"
        f.write(line)
with open("aggregated_svd_features_test" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
    users_offset = sup_info[0]
    line = "offset"
    for i in range(user_feature.shape[1]):
        line += ",feature_%d" % i
    movies_avg = sup_info[1]
    line += ",avg"
    for i in range(movie_feature.shape[1]):
        line += ",feature_%d" % i
    line += "\n"
    f.write(line)
    for el in test:
        user = el[0] - 1
        movie = el[1] - 1
        line = "%f" % users_offset[user]
        for i in range(user_feature.shape[1]):
            line += ",%f" % user_feature[user, i]
        line += ",%f" % movies_avg[movie]
        for i in range(movie_feature.shape[1]):
            line += ",%f" % movie_feature[movie, i]
        line += "\n"
        f.write(line)
