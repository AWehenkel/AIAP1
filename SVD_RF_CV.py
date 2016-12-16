import pandas as pd
import numpy as np
import time
import CollaborativeFiltering2
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sparse
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score



train = pd.read_csv("data/data_train.csv", delimiter=",").values
train_values = pd.read_csv("data/output_train.csv", delimiter=",").values
train = np.append(train, train_values, axis=1)
users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",").values
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",").values
np.random.shuffle(train)

num_components = 40  # number of components

nb_fold = 10
nb_data = train.shape[0]
fold_size = round(nb_data / nb_fold - 0.5)
score = []
for fold in range(nb_fold):

    start = fold * fold_size
    end = min(nb_data, (fold + 1) * fold_size)
    test = train[start:end, :]
    if start != 0:
        train_subset = np.append(train[:start, :], train[end:, :], axis=0)
    else:
        train_subset = train[end:, :]

    #train_subset = train
    row = train_subset[:, 0] - 1
    col = train_subset[:, 1] - 1
    data = train_subset[:, 2]

    #Average rating by movie and offset by user
    print("Average rating by movie and offset by user")
    users_offset = np.zeros(users.shape[0])
    users_nb_movie = np.zeros(users.shape[0])
    movies_avg = np.zeros(movies.shape[0])
    movies_nb_user = np.zeros(movies.shape[0])
    rating_matrix = np.zeros((users.shape[0], movies.shape[0]))
    # Creation des matrices offset et mean
    for el in train_subset[:]:
        users_nb_movie[el[0] - 1] += 1
        users_offset[el[0] - 1] += el[2]
        movies_nb_user[el[1] - 1] += 1
        movies_avg[el[1] - 1] += el[2]

    k_movie = 20
    global_avg_movie = movies_avg.sum() / movies_nb_user.sum()

    k_user = 20
    global_avg_user = users_offset.sum() / users_nb_movie.sum()
    movies_avg = np.divide((movies_avg + k_movie * global_avg_movie), (movies_nb_user + k_movie))

    users_offset = np.divide((users_offset + k_user * global_avg_user), (k_user + users_nb_movie)) - movies_avg.mean()

    #SVD Features
    print("SVD feature creation")
    known_rating = sparse.csr_matrix((data, (row, col)), shape=(users.shape[0], movies.shape[0]))
    SVD = TruncatedSVD(n_components=num_components, n_iter=100)
    user_feature = SVD.fit_transform(known_rating)
    movie_feature = np.transpose(SVD.components_)

    #Features for the learning subset
    print("Feature for the learning subset")
    aggregated_feature_train = np.zeros((train_subset.shape[0], 2 * num_components + 2 + users.shape[1] + movies.shape[1]))
    for i in range(train_subset.shape[0]):
        user_id = train_subset[i, 0] - 1
        movie_id = train_subset[i, 1] - 1
        aggregated_feature_train[i, :2 * num_components] = np.append(user_feature[user_id, :], movie_feature[movie_id, :])
        aggregated_feature_train[i, 2 * num_components] = users_offset[user_id]
        aggregated_feature_train[i, 2 * num_components + 1] = movies_avg[movie_id]
        aggregated_feature_train[i, (2 * num_components + 2):] = np.append(users[user_id, :], movies[movie_id, :])


    #Features for the testing subset
    print("Feature for the testing subset")
    aggregated_feature_test = np.zeros((test.shape[0], 2 * num_components + 2 + users.shape[1] + movies.shape[1]))
    for i in range(test.shape[0]):
        user_id = test[i, 0] - 1
        movie_id = test[i, 1] - 1
        aggregated_feature_test[i, :2 * num_components] = np.append(user_feature[user_id, :], movie_feature[movie_id, :])
        aggregated_feature_test[i, 2 * num_components] = users_offset[user_id]
        aggregated_feature_test[i, 2 * num_components + 1] = movies_avg[movie_id]
        aggregated_feature_test[i, 2 * num_components + 2:] = np.append(users[user_id, :], movies[movie_id, :])

    #Random forest training
    print("Random forest fitting")
    bag1 = ens.RandomForestRegressor(n_estimators=1000, max_depth=50, n_jobs=8, verbose=True, max_features="log2")
    #bag1 = ens.ExtraTreesRegressor(n_estimators=1000, max_depth=45, n_jobs=8, max_features="log2")
    #bag1 = ens.RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=8, verbose=True, max_features="log2")
    bag1.fit(aggregated_feature_train, train_subset[:, 2])
    # Score on testing subset
    print("Testing without predicted values for svd...")
    predictions = bag1.predict(aggregated_feature_test)
    mse = ((predictions - test[:, 2]) ** 2).mean()
    score.append(mse)
    print("Result: %f" % mse)
    loop_svd = True
    if(loop_svd):
        predicted_ratings = np.zeros((users.shape[0], movies.shape[0]))
        not_rated = []
        to_predict = []
        for user_id in range(users.shape[0]):
            for movie_id in range(movies.shape[0]):
                if(rating_matrix[user_id, movie_id] == 0):
                    features = np.append(user_feature[user_id, :], movie_feature[movie_id, :])
                    features = np.append(features, users_offset[user_id])
                    features = np.append(features, movies_avg[movie_id])
                    features = np.append(features, np.append(users[user_id, :], movies[movie_id, :]))
                    not_rated.append([user_id, movie_id])
                    to_predict.append(features)
                else:
                    predicted_ratings[user_id, movie_id] = rating_matrix[user_id, movie_id]
        predicted = bag1.predict(to_predict)
        i = 0
        for user_id in range(users.shape[0]):
            for movie_id in range(movies.shape[0]):
                if (rating_matrix[user_id, movie_id] == 0):
                    predicted_ratings[user_id, movie_id] = predicted[i]
                    i += 1
                else:
                    predicted_ratings[user_id, movie_id] = rating_matrix[user_id, movie_id]
        print()
        # SVD Features
        print("SVD feature creation")
        known_rating = predicted_ratings
        SVD = TruncatedSVD(n_components=num_components, n_iter=100)
        user_feature = SVD.fit_transform(known_rating)
        movie_feature = np.transpose(SVD.components_)
        # Features for the learning subset
        print("Feature for the learning subset")
        aggregated_feature_train = np.zeros(
            (train_subset.shape[0], 2 * num_components + 2 + users.shape[1] + movies.shape[1]))
        for i in range(train_subset.shape[0]):
            user_id = train_subset[i, 0] - 1
            movie_id = train_subset[i, 1] - 1
            aggregated_feature_train[i, :2 * num_components] = np.append(user_feature[user_id, :],
                                                                         movie_feature[movie_id, :])
            aggregated_feature_train[i, 2 * num_components] = users_offset[user_id]
            aggregated_feature_train[i, 2 * num_components + 1] = movies_avg[movie_id]
            aggregated_feature_train[i, (2 * num_components + 2):] = np.append(users[user_id, :], movies[movie_id, :])

        # Features for the testing subset
        print("Feature for the testing subset")
        aggregated_feature_test = np.zeros((test.shape[0], 2 * num_components + 2 + users.shape[1] + movies.shape[1]))
        for i in range(test.shape[0]):
            user_id = test[i, 0] - 1
            movie_id = test[i, 1] - 1
            aggregated_feature_test[i, :2 * num_components] = np.append(user_feature[user_id, :],
                                                                        movie_feature[movie_id, :])
            aggregated_feature_test[i, 2 * num_components] = users_offset[user_id]
            aggregated_feature_test[i, 2 * num_components + 1] = movies_avg[movie_id]
            aggregated_feature_test[i, 2 * num_components + 2:] = np.append(users[user_id, :], movies[movie_id, :])

        print("Random forest fitting")
        bag1 = ens.RandomForestRegressor(n_estimators=1000, max_depth=50, n_jobs=8, verbose=True, max_features="log2")
        # bag1 = ens.ExtraTreesRegressor(n_estimators=1000, max_depth=45, n_jobs=8, max_features="log2")
        # bag1 = ens.RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=8, verbose=True, max_features="log2")
        bag1.fit(aggregated_feature_train, train_subset[:, 2])

        #Score on testing subset
        print("Testing with better svd values...")
        predictions = bag1.predict(aggregated_feature_test)
        mse = ((predictions - test[:, 2])**2).mean()
        score.append(mse)
        print("Result: %f" % mse)

print(score)

