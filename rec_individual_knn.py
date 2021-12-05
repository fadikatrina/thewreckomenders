from random import randrange

from IPython.core.display import display
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import helpers_data_load as dl
import expl_knn

df_recipes = None


def knn(users_df, recipe_df, selected_user):
    rated_recipes_df, unrated_recipes_df = prepare_data(users_df, recipe_df, selected_user)
    x = rated_recipes_df.drop('rating', axis=1)
    y = rated_recipes_df['rating']
    x_unrated = unrated_recipes_df

    neigh = KNeighborsRegressor(n_neighbors=6)

    # print(x)
    # print(y)
    neigh.fit(x, y)

    y_unrated = neigh.predict(x_unrated)
    unrated_recipes_df['predicted_ratings_KNN'] = y_unrated
    unrated_recipes_df = unrated_recipes_df.sort_values(by='predicted_ratings_KNN', ascending=False)
    # display(unrated_recipes_df.head())
    row = dl.recipes_raw.loc[dl.recipes_raw.index[unrated_recipes_df.index[0] + 1]]
    expl_knn.indiv_CB(str(row['name']), str(unrated_recipes_df['predicted_ratings_KNN'].iloc[0]))


def prepare_data(users, recipes, selected_user):
    recipes.index.name = 'item'

    selected_user_ratings = users.loc[users['user'] == selected_user]
    selected_user_ratings = selected_user_ratings.sort_values(by='item', ascending=True)

    # print("Rated recipes: " + str(selected_user_ratings.shape[0]))
    recipes['minutes'] = min_max_scaling(recipes['minutes'])
    recipes['n_steps'] = min_max_scaling(recipes['n_steps'])
    recipes['n_ingredients'] = min_max_scaling(recipes['n_ingredients'])
    rated_recipes_df = recipes.loc[list(selected_user_ratings['item'])]
    rated_recipes_df = rated_recipes_df[['minutes']]
    rated_recipes_df = rated_recipes_df.join(selected_user_ratings.set_index('item')['rating'], on='item')

    diff = set(recipes.index) - set(rated_recipes_df.index)
    unrated_recipes_df = recipes.loc[diff]
    unrated_recipes_df = unrated_recipes_df[['minutes']]

    return rated_recipes_df, unrated_recipes_df


def train_test_holdout(users, recipes, selectedUser):
    rated = prepare_data(users, recipes, selectedUser)[0]

    train, test = train_test_split(rated, test_size=0.1)

    x_train = train.drop('rating', axis=1)
    y_train = train['rating']

    x_test = test.drop('rating', axis=1)
    y_test = test['rating']

    neigh = KNeighborsRegressor(n_neighbors=12)
    neigh.fit(x_train, y_train)  # train our classifier

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

    # print(y_test)
    # print(y_pred)
    # print(relevant_test)
    # print(relevant_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(relevant_test, relevant_pred,
                                                                   average="binary", zero_division=0)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r_squared = r2_score(relevant_test, relevant_pred)

    return precision, recall, fscore, rmse, r_squared


def holdout(users, recipes):
    user_list = list(set(users['user']))
    precision_list = []
    recall_list = []
    fscore_list = []
    rmse_list = []
    r2_list = []
    counter = 0

    for user in user_list:
        selected_user_ratings = users.loc[users['user'] == user]

        if len(selected_user_ratings) >= 150:
            counter += 1
            precision, recall, fscore, rmse, r2 = train_test_holdout(users, recipes, user)
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)
            rmse_list.append(rmse)
            r2_list.append(r2)

    print('Precision', np.mean(precision_list))
    print('Recall', np.mean(recall_list))
    print('Fscore', np.mean(fscore_list))
    print('RMSE', np.mean(rmse_list))
    print('R-squared', np.mean(r2_list))
    print('Evaluated on', counter, 'users')


def min_max_scaling(column):
    max_val = column.max()
    min_val = column.min()
    column = (column - min_val) / (max_val - min_val)
    return column


dl.load_recipes_raw()
knn(dl.df_users, dl.recipes_raw, 555)