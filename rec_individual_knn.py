from random import randrange

from IPython.core.display import display
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from helpers_data_load import df_users, recipe_data
from rec_individual_svd import getVectorizedData, svd
import expl_content_based
from logger import l

df_recipes = None


# todo:high add explanations
# todo: use SVD
# todo: experimenting the kNN with
# - different k
# - different size of sample
# - different features

def knn(users_df, recipe_df, selected_user, svdFlag):
    rated_recipes_df, unrated_recipes_df = prepare_data(users_df, recipe_df, False, selected_user)
    x, vectorizer = getVectorizedData(rated_recipes_df)
    x_unrated = getVectorizedData(unrated_recipes_df)[0]
    y = rated_recipes_df['rating']

    components = len(rated_recipes_df)

    if components > 100:
        components = 100

    if svdFlag:
        x = svd(x, vectorizer, components)
        x_unrated = svd(x_unrated, vectorizer, components)
    else:
        x = vectorizer.fit_transform(x['tags'] + x['steps'] + x['ingredients'])
        x_unrated = vectorizer.transform(x_unrated['tags'] + x_unrated['steps'] + x_unrated['ingredients'])

    neigh = KNeighborsRegressor(n_neighbors=5)

    print(x)
    print(y)
    neigh.fit(x, y)

    y_unrated = neigh.predict(x_unrated)
    unrated_recipes_df['predicted_ratings_KNN'] = y_unrated
    unrated_recipes_df = unrated_recipes_df.sort_values(by='predicted_ratings_KNN', ascending=False)
    display(unrated_recipes_df.head())


def prepare_data(users, recipes, selected_user):
    recipes.index.name = 'item'

    selected_user_ratings = users.loc[users['user'] == selected_user]
    selected_user_ratings = selected_user_ratings.sort_values(by='item', ascending=True)

    print("Rated recipes: " + str(selected_user_ratings.shape[0]))

    rated_recipes_df = recipes.loc[list(selected_user_ratings['item'])]
    rated_recipes_df = rated_recipes_df[['tags', 'steps', 'ingredients']]
    rated_recipes_df = rated_recipes_df.join(selected_user_ratings.set_index('item')['rating'], on='item')

    diff = set(recipes.index) - set(rated_recipes_df.index)
    unrated_recipes_df = recipes.loc[diff]
    unrated_recipes_df = unrated_recipes_df[['tags', 'steps', 'ingredients']]

    return rated_recipes_df, unrated_recipes_df


def train_test_holdout(users, recipes, selectedUser, svdFlag):
    rated = prepare_data(users, recipes, selectedUser)[0]

    x, vectorizer = getVectorizedData(rated)
    y = rated['rating']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    components = len(y)
    if components > 75:
        components = 75

    if svdFlag:
        x_train = svd(x_train, vectorizer, components)
        x_test = svd(x_test, vectorizer, components)
    else:
        x_train = vectorizer.fit_transform(x_train['tags'] + x_train['steps'] + x_train['ingredients'])
        x_test = vectorizer.transform(x_test['tags'] + x_test['steps'] + x_test['ingredients'])

    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(x_train, y_train)  # train our cassifier

    y_pred = neigh.predict(x_test)  # evaluates the predictions of the classifier
    relevant_test = []
    relevant_pred = []

    for i in range(len(y_test)):
        if y_test.to_numpy()[i] > 3:
            relevant_test.append(1)
        else:
            relevant_test.append(0)
        if y_pred[i] > 3:
            relevant_pred.append(1)
        else:
            relevant_pred.append(0)

    print(y_test)
    print(y_pred)
    print(relevant_test)
    print(relevant_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(relevant_test, relevant_pred,
                                                                   average="binary", zero_division=0)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return precision, recall, fscore, rmse


def holdout(users, recipes):
    user_list = list(set(users['user']))
    precision_list = []
    recall_list = []
    fscore_list = []
    rmse_list = []

    for user in user_list:
        selected_user_ratings = users.loc[users['user'] == user]

        if len(selected_user_ratings) > 100:
            precision, recall, fscore, rmse = train_test_holdout(users, recipes, user, False)
            if precision > 0:
                precision_list.append(precision)
            if recall > 0:
                recall_list.append(recall)
            if fscore > 0:
                fscore_list.append(fscore)
            if rmse > 0:
                rmse_list.append(rmse)

    print('Precision', np.mean(precision_list))
    print('Recall', np.mean(recall_list))
    print('Fscore', np.mean(fscore_list))
    print('RMSE', np.mean(rmse_list))


print(holdout(df_users, recipe_data))
# print(train_test_holdout(df_users, recipe_data, 555, False))
