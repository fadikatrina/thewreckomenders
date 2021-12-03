from random import randrange

from IPython.core.display import display
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

def knn(df_users, recipe_data, selected_user, svdFlag):
    recipe_data.index.name = 'item'

    selected_user_ratings = df_users.loc[df_users['user'] == selected_user]
    selected_user_ratings = selected_user_ratings.sort_values(by='item', ascending=True)

    print("Rated movies: " + str(selected_user_ratings.shape[0]))

    rated_recipes_df = recipe_data.loc[list(selected_user_ratings['item'])]
    rated_recipes_df = rated_recipes_df[['tags', 'steps', 'ingredients']]
    rated_recipes_df = rated_recipes_df.join(selected_user_ratings.set_index('item')['rating'], on='item')

    diff = set(recipe_data.index) - set(rated_recipes_df.index)
    unrated_recipes_df = recipe_data.loc[diff]
    unrated_recipes_df = unrated_recipes_df[['tags', 'steps', 'ingredients']]

    x, vectorizer = getVectorizedData(rated_recipes_df)
    x_unrated = getVectorizedData(unrated_recipes_df)[0]
    y = rated_recipes_df['rating']

    components = selected_user_ratings.shape[0]

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


knn(df_users, recipe_data, 714, False)
knn(df_users, recipe_data, 714, True)
