import numpy as np
import pandas as pd
import sklearn.tree as tree
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score
import sklearn.utils as util
from sklearn.tree import DecisionTreeRegressor
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
util.check_random_state(1)

features = pd.read_csv('agregated_data_train_11-12-2016_15h39.csv', delimiter=',')
#features = pd.read_csv('aggregated_svd_features_train_11-12-2016_17h36.csv', delimiter=',')
output = pd.read_csv('data/output_train.csv', delimiter=',')

#features = np.append(features1, features, axis=1)

t_o = np.append(features, output, axis=1)
nb_test = 0
scores = []
for i in range(nb_test):
    np.random.shuffle(t_o)

    x = t_o[:, :-1]
    y = t_o[:, -1]

    bag1 = ens.ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=8)
    bag1.fit(x[:-13000], y[:-13000])

    predicted = bag1.predict(x[-13000:])

    score = ((predicted - y[-13000:])** 2).mean()

    print(score)

    scores.append(score)

print("Average score: %f" % np.mean(scores))

np.random.shuffle(t_o)
x = t_o[:, :-1]
y = t_o[:, -1]
#bag1 = ens.RandomForestRegressor(n_estimators=3000, n_jobs=8, verbose=True, max_features="log2")
#bag1 = ens.AdaBoostRegressor(n_estimators=100, base_estimator=DecisionTreeRegressor(max_depth=3), loss='exponential')
bag1 = ens.ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=8)
bag1.fit(x, y)
nb_fold = 10
print(np.mean(cross_val_score(bag1, x, y, cv=nb_fold, scoring='neg_mean_squared_error', verbose=10)))
print("ok")
#to_predict_feature = pd.read_csv("aggregated_svd_features_test_11-12-2016_17h36.csv",  delimiter=',').values
#to_predict_feature1 = pd.read_csv("agregated_data_test_11-12-2016_15h58.csv",  delimiter=',').values
#to_predict_feature = np.append(to_predict_feature1, to_predict_feature, axis=1)
#test = pd.read_csv("data/data_test.csv",  delimiter=',').values
#predicted = bag1.predict(to_predict_feature1)
#make_submission(predicted, test[:, 0], test[:, 1], name="probably_overfitted")
