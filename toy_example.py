#! /usr/bin/env python
# -*- coding: utf-8 -*-


## Import packages
import os
import numpy as np
import pandas as pd
import datetime
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
            line = '{:0.0f},{:0.0f},{}\n'.format(user_id_test[i],movie_id_test[i],y_predict[i])
            f.write(line)
    print("Submission file successfully written!")

def userDataNormalize(data):
    data_size = data.shape[0]
    user_id = np.zeros(data_size)
    age = np.zeros(data_size)
    inv_nb_age_cat = 1
    genders = np.zeros(data_size)
    occupations = []
    zip_codes = np.zeros(data_size)
    len_code = 2

    for i in range(data_size):
        user_id[i] = int(data.iloc[i]["user_id"])
        #Divide the age by a bigger number if we want fewer age categories
        age[i] = int(data.iloc[i]["age"]/inv_nb_age_cat)
        #Store M as 0 and F as 1
        gender = data.iloc[i]["gender"]
        if(gender == "F"):
            genders[i] = 1
        #Create a list of all occupations
        occupation = data.iloc[i]["occupation"]
        if occupation not in occupations:
            occupations.append(occupation)
        #Store only the len_code first digits of the zip_code
        zip_code = data.iloc[i]["zip_code"]
        min_len = min([len_code, len(zip_code)])
        zip_code_begin = zip_code[0:len_code]
        if(zip_code_begin[0].isdigit()):
            zip_codes[i] = int(zip_code_begin.lstrip())
        else:
            zip_codes[i] = 0 #Je ne sais pas trop quoi faire des zip_code qui contiennent des lettres ...

    with open("user_data_normalized" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        f.write('"user_id","age","gender","zip_code"')
        for occupation in occupations:
            f.write(',"%s"' % occupation)
        f.write('\n')

        for i in range(data_size):
            line = '{:0.0f},{:0.0f},{:0.0f}'.format(user_id[i], age[i], genders[i])
            for occupation in occupations:
                if(occupation == data.iloc[i]["occupation"]):
                    line += ',{:0.0f}'.format(1)
                else:
                    line += ',{:0.0f}'.format(0)
            line += '\n'
            f.write(line)

def movieDataNormalizer(data):
    data_size = data.shape[0]
    year = np.zeros(data_size)
    date = np.zeros(data_size)
    movie_type_cast_int = np.zeros(data_size)
    movie_id_available = np.zeros(data_size)
    movie_title_size = np.zeros(data_size)
    words = {}
    nb_word = 20
    most_used_word = ["" for i in range(nb_word)]
    most_used_word_number = [0 for i in range(nb_word)]


    for i in range(data_size):
        print(i)
        if(data.loc[i]["unknown"] == 0):
            for word in data.loc[i]["movie_title"].replace(",", "").split("(")[0].split():
                if(word[0] != "("):
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1
            movie_title_size[i] = len(data.loc[i]["movie_title"])
            movie_id_available[i] = i + 1
            year[i] = datetime.datetime.strptime(data.loc[i]["release_date"], "%d-%b-%Y").year
            date[i] = time.mktime(datetime.datetime.strptime(data.loc[i]["release_date"], "%d-%b-%Y").timetuple())
            for j in range(6, data.shape[1]):
                movie_type_cast_int[i] += 2^(j - 6)*data.iloc[i][j]
        else:
            movie_id_available[i] = -1
    date = (date - min(date))
    date = 1000*date/max(date)
    #normalized_data = np.matrix((h,4))
    #normalized_data = [movie_id_available[1:h], year[1:h], date[1:h], movie_type_cast_int[1:h]]

    #Compute the nb_word words the most frequent in the titles
    for key in words:
        if (words[key] > min(most_used_word_number) and len(key) > 2 and not(key in ("The", "and", "the", "For", "for", "with"))):
            id = most_used_word_number.index(min(most_used_word_number))
            most_used_word[id] = key
            most_used_word_number[id] = words[key]


    #Compute the characteristic of the title
    word_in_title = np.zeros((data_size, nb_word))
    for movie_id in range(data_size):
        for i in range(nb_word):
            if most_used_word[i] in data.loc[movie_id]["movie_title"]:
                word_in_title[movie_id][i] = 1
    print("ok")
    with open("movie_data_normalized" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        f.write('"MOVIE_ID","date_norm"')
        for movie_type in data.keys()[6:]:
            f.write(',"%s"' % movie_type)
        #for word in most_used_word:
         #   f.write(',"%s"' % word)
        f.write('\n')

        for i in range(data_size):
            line = '{:0.0f},{:0.3f}'.format(movie_id_available[i], date[i])
            for movie_type in data.keys()[6:]:
                line += ',{:0.0f}'.format(data.iloc[i][movie_type])
           # for j in range(nb_word):
            #    line += ',{:0.0f}'.format(word_in_title[i][j])
            line += '\n'
            f.write(line)
    #print(title)


def aggregateData(data, user, movie):
    with open("agregated_data_test" + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M')) + ".csv", 'w') as f:
        #for el in data.keys():
         #   f.write("%s," % el)
        for el in user.keys()[1:]:
            f.write('"%s",' % el)
        for el in movie.keys()[1:-1]:
            f.write('"%s",' % el)
        f.write('"%s"\n' % movie.keys()[-1])

        for i in range(data.shape[0]):
            line = '';
            #for el in data.iloc[i][:]:
             #   line += '{:0.0f},'.format(el)

            user_id = data.iloc[i]["user_id"]
            for el in user.iloc[user_id - 1][1:]:
                line += '{:0.0f},'.format(el)

            print(i)
            movie_id = data.loc[i]["movie_id"]
            for el in movie.iloc[movie_id - 1][1:-1]:
                line += '{:0.0f},'.format(el)

            line += '{:0.0f}\n'.format(movie.loc[movie_id - 1][-1])
            f.write(line)


if __name__ == "__main__":


    # Load data_train
    #data_train = pd.read_csv('data/data_train.csv', delimiter=',')
    # Load output_train
    #output_train = pd.read_csv('data/output_train.csv', delimiter=',')

    # Load data_test
    data_test = pd.read_csv('data/data_train.csv', delimiter=',')
    movie = pd.read_csv('data/movie_data_normalized_08-12-2016_22h16.csv', delimiter=',')
    user = pd.read_csv('data/user_data_normalized_07-12-2016_09h33.csv', delimiter=',')
    aggregateData(data_test, user, movie)

    # Load user info
    #data_user = pd.read_csv('data/data_user.csv', delimiter=',')
    #userDataNormalize(data_user)
    #print(data_user.query('occupation == "other"').query('gender == "M"'))
    # Load movie info
    #data_movie = pd.read_csv('data/data_movie.csv', delimiter=',', encoding="latin_1")
    #pd.read_cs
    #movieDataNormalizer(data_movie)
    # Create matrix "users x movies" for training data

    #n_users = len(np.unique(data_user['user_id']))
    #n_movies = len(np.unique(data_movie['movie_id']))

    #matrix = np.zeros((n_users, n_movies))

#    print("Building the matrix...")
    '''
    for i in range(data_train.shape[0]):
        user_idx = data_train.loc[i]['user_id']-1
        item_idx = data_train.loc[i]['movie_id']-1
        rating_idx = output_train.loc[i]['rating']
        matrix[int(user_idx), int(item_idx)] = int(rating_idx)    

    matrix[matrix==0] = np.nan

    # Toy prediction
    n_elements = data_test.shape[0]
    
    y_predict = np.ones((n_elements,)) * 5

    # Make a submission file

    #make_submission(y_predict, name="toy_submission", user_id_test=data_test['user_id'], movie_id_test=data_test['movie_id'])
    '''

