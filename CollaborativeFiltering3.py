import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import time
import CollaborativeFiltering2

def getSuppValuesByCat():
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
    users_offset, movies_avg = CollaborativeFiltering2.getSuppValues()
    k_movie = 20
    k_user = 20

    #Fill in the four principal matrices
    for el in train[:]:
        user_id = el[0]-1
        movie_id = el[1]-1
        rating = el[2]

        i = 0
        for works in users_work[user_id]:
            if works == 1:
                movies_view_per_work[movie_id,i] += 1
                movies_avg_per_work[movie_id,i] += rating
            i += 1

        i = 0
        for genre in movies_genre[movie_id]:
            if genre == 1:
                users_view_per_genre[user_id,i] += 1
                users_avg_per_genre[user_id,i] += rating
            i += 1


    movies_per_genre = users_view_per_genre.sum(axis=0) #!!! il y a certains films qui ont plusieurs genre...
    movies_genre_avg = np.divide(users_avg_per_genre.sum(axis=0),movies_per_genre)
    user_per_work = movies_view_per_work.sum(axis=0)
    user_work_avg = np.divide(movies_avg_per_work.sum(axis=0),user_per_work)

    #division a la main pour eviter les divisions par 0
    for i in range(users_avg_per_genre.shape[0]):
        for j in range(users_avg_per_genre.shape[1]):
            if(users_view_per_genre[i,j] != 0):
                users_avg_per_genre[i,j] /= users_view_per_genre[i,j]
    users_offset_per_genre = users_avg_per_genre - movies_genre_avg
    for i in range(users_offset_per_genre.shape[1]):
        users_offset_per_genre[:,i] = np.divide((users_offset_per_genre[:,i]+k_user*users_offset), 1 + k_user)

    #Est-ce qu'on moyenne avec movies_avg ou avec movie_genre_avg??
    for i in range(movies_avg_per_work.shape[1]):
        movies_avg_per_work[:,i] = np.divide((movies_avg_per_work[:,i] + k_movie*movies_avg), (movies_view_per_work[:,i]+k_movie))

    #user_tot_views = sum(users_view_per_genre, axis=)

    return (movies_avg_per_work, users_offset_per_genre)


movies_avg_per_work, users_offset_per_genre = getSuppValuesByCat()
total_size = movies_avg_per_work.shape[1] + users_offset_per_genre.shape[1]

# Creating the training csv for the output of this file
'''
train = pd.read_csv("data/data_train.csv", delimiter=",")
train = train.values
name = "train_aggregated_categories" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv"
with open(name, 'w') as f:
    for i in range(total_size-1):
        f.write('"feature_%d",' % i)
    f.write('"feature_%"\n' % (total_size-1))
    for el in train[:]:
        user_id = el[0]-1
        movie_id = el[1]-1
        for feat in users_offset_per_genre[user_id]:
            f.write('%f,' % feat)
        for i in range(movies_avg_per_work.shape[1]):
            f.write('%f,' % movies_avg_per_work[movie_id][i])
        f.write('%f\n' % movies_avg_per_work[movie_id][movies_avg_per_work.shape[1]-1])
'''

#Same for the test
'''
test = pd.read_csv("data/data_test.csv", delimiter=",")
test = test.values
name = "test_aggregated_categories" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv"
with open(name, 'w') as f:
    for i in range(total_size-1):
        f.write('"feature_%d",' % i)
    f.write('"feature_%d"\n', (total_size-1))
    for el in test[:]:
        user_id = el[0]-1
        movie_id = el[1]-1
        for feat in users_offset_per_genre[user_id]:
            f.write('%f,' % feat)
        for i in range(movies_avg_per_work.shape[1]):
            f.write('%f,' % movies_avg_per_work[movie_id][i])
        f.write('%f\n' % movies_avg_per_work[movie_id][movies_avg_per_work.shape[1]-1])
'''

#Creates the training csv with the original data and the data generated by this file
'''
name = "train_aggregated_cat_and_norm" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv"
train_aggregated_cat = pd.read_csv("train_aggregated_categories_12-12-2016_23h17.csv", delimiter=",")
keys_cat = train_aggregated_cat.keys()
val_cat = train_aggregated_cat.values
train_aggregated_norm = pd.read_csv("data/train_agregated_data_test_08-12-2016_22h26.csv", delimiter=",")
keys_norm = train_aggregated_norm.keys()
keys_norm = keys_norm.append(keys_cat)
val_norm = train_aggregated_norm.values
val = []
for i in range(val_norm.shape[0]):
    val.append(np.append(val_norm[i,:], val_cat[i,:]))
val_array = np.asarray(val)
df = pd.DataFrame(data=val, columns=keys_norm)
df.to_csv(name)
'''

#Same for the test
'''
name = "test_aggregated_cat_and_norm" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv"
test_aggregated_cat = pd.read_csv("test_aggregated_categories_12-12-2016_23h23.csv", delimiter=",")
keys_cat = test_aggregated_cat.keys()
val_cat = test_aggregated_cat.values
train_aggregated_norm = pd.read_csv("data/test_agregated_data_test_08-12-2016_22h18.csv", delimiter=",")
keys_norm = train_aggregated_norm.keys()
keys_norm = keys_norm.append(keys_cat)
val_norm = train_aggregated_norm.values
val = []
for i in range(val_norm.shape[0]):
    val.append(np.append(val_norm[i,:], val_cat[i,:]))
val_array = np.asarray(val)
df = pd.DataFrame(data=val, columns=keys_norm)
df.to_csv(name)
'''

#Compute a rating matrix for each user and film
'''
users = pd.read_csv("data/user_data_normalized_28-11-2016_01h32.csv", delimiter=",")
movies = pd.read_csv("data/movie_data_normalized.csv", delimiter=",")
movies_genre = movies.loc[:,"Action":"Western"].values #gives for each film its genre
users_work = users.loc[:,"other":"none"].values #gives for each user its occupation
users_offset, movies_avg = CollaborativeFiltering2.getSuppValues()
rating_matrix = np.zeros((users.shape[0], movies.shape[0]))

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
        #some users might have several works or no work
        if nb_work != 0:
            work_avg /= nb_work
        else:
            work_avg = np.mean(movies_avg)

        #get average for the film genre
        i = 0
        genre_off = 0
        nb_genre = 0
        for genre in movies_genre[movie]:
            if genre == 1:
                genre_off += users_offset_per_genre[user, i]
                nb_genre += 1
            i += 1
        #some films might have several genres or no genres
        if nb_genre != 0:
            genre_off /= nb_genre
        else:
            genre_off = np.mean(users_offset)

        rating_matrix[user, movie] = work_avg + genre_off

print(rating_matrix)
'''