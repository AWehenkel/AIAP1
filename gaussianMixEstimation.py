import sklearn.cluster as clu
import numpy as np
import pandas as pd
import math
import time

class GaussianMixEstimation:
    def groupObjectFeature(self, x, y, nb_group):
        if(x.shape[1] == 1 and x.shape == y.shape):
            x = (x - min(x))/(max(x) - min(x))
            y = (y - min(y))/(max(y) - min(y))
            data = np.zeros((x.shape[0], 2))
            data[:][0] = x[:][0]
            data[:][1] = y[:][0]
            km = clu.KMeans(nb_group, verbose=1)

    def computeGaussianMatrix(self, row_id, col_id, col_row_aggregate):
        nb_row = len(row_id)
        nb_col = len(col_id)

        sigma = np.zeros((nb_row, nb_col))
        mu = np.zeros((nb_row, nb_col))
        row_name = col_row_aggregate.keys()[0]
        col_name = col_row_aggregate.keys()[1]
        for i in range(nb_row):
            query = "%s == %s" % (row_name, row_id[i])
            col = col_row_aggregate.query(query)
            for j in range(nb_col):
                query = "%s == %s" % (col_name, col_id[j])
                val = col.query(query)
                print(val.values)


n = GaussianMixEstimation()
row_id = (10, 15, 20)
col_id = (34, 25, 10)
data_user = pd.read_csv('data/data_user.csv', delimiter=',')
job = ("Man","Woman","other","executive","writer","administrator","student","marketing","educator","librarian","entertainment","engineer","programmer","scientist","artist","lawyer","retired","salesman","homemaker","technician","doctor","healthcare")
movie_types = ("Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western")
#n.computeGaussianMatrix(row_id, col_id, data_user)
data = pd.read_csv('agregated_data_28-11-2016_01h50.csv', delimiter=',')
ok = data.query('gender == 0').query('Action == 1')
ids = ok["gender"].index.tolist()
rating = pd.read_csv('data/output_train.csv', delimiter=',')
'''
mu_frame = pd.DataFrame(columns=job,index=movie_types)
std_frame = pd.DataFrame(columns=job,index=movie_types)

for movie_type in movie_types:
    ok = data.query('gender == 0').query('%s == 1' % movie_type)
    ids = ok["gender"].index.tolist()
    std_frame.set_value(movie_type,"Man",rating.ix[ids].values.std())
    mu_frame.set_value(movie_type,"Man",rating.ix[ids].values.mean())

for movie_type in movie_types:
    ok = data.query('gender == 1').query('%s == 1' % movie_type)
    ids = ok["gender"].index.tolist()
    std_frame.set_value(movie_type,"Woman",rating.ix[ids].values.std())
    mu_frame.set_value(movie_type,"Woman",rating.ix[ids].values.mean())

for activity in job[2:]:
    for movie_type in movie_types:
        ok = data.query('%s == 1' % activity).query('%s == 1' % movie_type)
        ids = ok["gender"].index.tolist()
        std_frame.set_value(movie_type, activity, rating.ix[ids].values.std())
        mu_frame.set_value(movie_type, activity, rating.ix[ids].values.mean())


mu_frame.to_csv("mu.csv")
std_frame.to_csv("std.csv")
'''
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

mu = pd.read_csv("mu.csv")
sigma = pd.read_csv("std.csv")
test = pd.read_csv("agregated_data_test_28-11-2016_11h11.csv", delimiter=",")
print(test.values.shape[0])
y = []
for sample in range(test.values.shape[0]):
    da = test.loc[sample][:]
    s = []
    m = []
    if da["gender"] == 0:
        sex = "Man"
    else:
        sex = "Woman"
    for movie_type in range(len(movie_types)):
        if da[movie_types[movie_type]] == 1:
            if not(math.isnan(mu.get_value(movie_type, sex))):
                s.append(sigma.get_value(movie_type, sex))
                m.append(mu.get_value(movie_type, sex))
            for activity in job[2:]:
                if da[activity] == 1:
                    if not (math.isnan(mu.get_value(movie_type, activity))):
                        s.append(sigma.get_value(movie_type, activity))
                        m.append(mu.get_value(movie_type, activity))

    mean_val = np.mean(m)
    if(math.isnan(mean_val)):
        mean_val = 3.5
    y.append(mean_val)


test_set_id = pd.read_csv('data/data_test.csv', delimiter=',')
make_submission(y, test_set_id["user_id"], test_set_id["movie_id"], "gaussian")