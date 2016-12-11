import numpy as np
import pandas as pd
import sklearn.tree as tree
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score
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

#features = pd.read_csv('agregated_data_train_11-12-2016_15h39.csv', delimiter=',')
features = pd.read_csv('aggregated_svd_features_train_11-12-2016_17h36.csv', delimiter=',')
output = pd.read_csv('data/output_train.csv', delimiter=',')



t_o = np.append(features, output, axis=1)
nb_test = 0
scores = []
for i in range(nb_test):
    np.random.shuffle(t_o)

    x = t_o[:, :-1]
    y = t_o[:, -1]

    bag1 = ens.RandomForestRegressor(n_estimators=50, n_jobs=8)
    bag1.fit(x[:-10000], y[:-10000])

    predicted = bag1.predict(x[-10000:])

    score = ((predicted - y[-10000:])** 2).mean()

    print(score)

    scores.append(score)

print("Average score: %f" % np.mean(scores))

np.random.shuffle(t_o)
x = t_o[:, :-1]
y = t_o[:, -1]
bag1 = ens.RandomForestRegressor(n_estimators=150, n_jobs=8, max_depth=10)
#bag1 = ens.ExtraTreesRegressor(n_estimators=150, max_depth=10)
bag1.fit(x, y)
#nb_fold = 10
#print(np.mean(cross_val_score(bag1, x, y, cv=nb_fold, scoring='neg_mean_squared_error', verbose=10)))
#print("ok")
to_predict_feature = pd.read_csv("aggregated_svd_features_test_11-12-2016_17h36.csv",  delimiter=',').values
test = pd.read_csv("data/data_test.csv",  delimiter=',').values
predicted = bag1.predict(to_predict_feature)
make_submission(predicted, test[:, 0], test[:, 1], name="probably_overfitted")
