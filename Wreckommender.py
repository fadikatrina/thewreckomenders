#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from ast import literal_eval
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# We load in our datasets, consisting of user data and data for the recipes. We will be using the ratings that users gave in combination with the recipe's ingredients, techniques and calories to perform recommendations


df_users = pd.read_csv('Data/PP_users.csv')
#Note that there are no duplicates
df_users.head()

# u = user_id,
# techniques = techniques used for items  that were interacted with(index is a technique with the number being a counter),
# items = item_ids of items that were interacted with,
# n_items = number of items reviewed,
# ratings = ratings for items reviewed,
# n_ratings = number of ratings

number_of_users_with_less_than_10_reviews = df_users[df_users["n_ratings"] < 10].shape[0]
number_of_users_with_less_than_5_reviews = df_users[df_users["n_ratings"] < 5].shape[0]
number_of_users = len(df_users.index)

print(str(round((number_of_users_with_less_than_10_reviews / number_of_users) * 100,
                1)) + "% of user have less than 10 reviews")
print(str(round((number_of_users_with_less_than_5_reviews / number_of_users) * 100,
                1)) + "% of user have less than 5 reviews")



df_recipes = pd.read_csv('Data/PP_recipes.csv')
#Note that there are no duplicates
df_recipes.head()

# id = recipe_id, i = Recipe ID mapped to contiguous integers from 0,
# name_tokes = BPE-tokenized recipe name,
# ingredient_tokens = BPE-tokenized ingredients list (list of lists),
# steps_tokens = BPE-tokenized steps,
# techniques = List of techniques used in recipe,
# calorie_level = either a 0, 1 or 2 indicating how much calories it contains,
# ingredient_ids = the ids of the ingredients used





df_users = pd.read_csv('Data/PP_users.csv')
df_users.drop('techniques', axis=1, inplace=True)
df_users.drop('n_items', axis=1, inplace=True)
df_users.drop('n_ratings', axis=1, inplace=True)

df_users = df_users.rename(columns={'u': 'user', 'items': 'item', 'ratings': 'rating'})

df_users.head()





# Needed to make the explode function, source: https://stackoverflow.com/questions/63472664/pandas-explode-function-not-working-for-list-of-string-column
df_users['rating'] = df_users['rating'].apply(literal_eval)
df_users['item'] = df_users['item'].apply(literal_eval)

df_users = df_users.explode(['rating', 'item'], ignore_index=True)
df_users.head()





df_recipes.drop('i', axis=1, inplace=True)
df_recipes.drop('name_tokens', axis=1, inplace=True)
df_recipes.drop('ingredient_tokens', axis=1, inplace=True)
df_recipes.drop('steps_tokens', axis=1, inplace=True)

df_recipes.head()





# Needed to make the explode function, source: https://stackoverflow.com/questions/63472664/pandas-explode-function-not-working-for-list-of-string-column
df_recipes['techniques'] = df_recipes['techniques'].apply(literal_eval)
df_recipes['ingredient_ids'] = df_recipes['ingredient_ids'].apply(literal_eval)

df_recipes = df_recipes.explode('techniques')
df_recipes = df_recipes.explode('ingredient_ids')
df_recipes.head()





# Where are we removing users and recipes with few ratings?
# Do we still need the explode for anything?
# Should we change the features that we're using?


# ## Health Filter




# Note that this file is made by us to only have to load nutritional info instead of a giant csv file
nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';')
nutrition_data.head()


# We want to give the user a (tweakable) health filter, so they can filter their suggestions to only contain healthy recipes.




# This functions returns the recipe_ids
# Note: limits takes max sugar amount, max sodium, min protein amount, max saturated_fat (not in grams, but percentage of nutritional content)
def ApplyHealthFilter(recipe_ids, limits=[.15, .35, .10, .25], debug_prints=False):
    healthy_recipes = []

    # Get the nutritional information for the relevant recipes
    recipes = nutrition_data.loc[nutrition_data['id'].isin(recipe_ids)]

    for index, recipe in recipes.iterrows():
        # Nutrition information in calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates
        nutrition_values = recipe['nutrition']
        # convert string version of array into a proper array
        nutrition_values = literal_eval(nutrition_values)

        sugar = nutrition_values[2]
        sodium = nutrition_values[3]
        protein = nutrition_values[4]
        saturated_fat = nutrition_values[5]

        # Since the nutritional info is in absolute numbers instead of per 100 grams, we'll normalize
        normalization_factor = sum(nutrition_values[1:])
        normalization_factor = max(normalization_factor, 0.01)

        sugar /= normalization_factor
        sodium /= normalization_factor
        protein /= normalization_factor
        saturated_fat /= normalization_factor

        if sugar < limits[0] and sodium < limits[1] and protein > limits[2] and saturated_fat < limits[3]:
            healthy_recipes.append(recipe['id'])

            if debug_prints:
                print("Healthy: ", index)

        else:
            if debug_prints:
                print("Unhealthy: ", index, sugar, sodium, protein, saturated_fat)

    return healthy_recipes


df_random = df_users.sample(n=20)
recipe_ids = df_random['item']
# limits = [.15, .35, .10, .25]

healhty_recipes = ApplyHealthFilter(recipe_ids)
print(healhty_recipes) # prints healthy recipes ids


# Since we have a lot of high dimensional data, we could make use of SVD to speed up computation.

# In[3]:


recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')
recipe_data.head()


# In[24]:


minutes = recipe_data['minutes']
tags = recipe_data['tags']
nutrition = recipe_data['nutrition']
steps = recipe_data['steps']
ingredients = recipe_data['ingredients']


# In[27]:


# Transform text into numerical data
vectorizer = TfidfVectorizer(max_features=10000)
tags = vectorizer.fit_transform(tags)
steps = vectorizer.fit_transform(steps)
ingredients = vectorizer.fit_transform(ingredients)

print(tags.shape)
print(steps.shape)
print(ingredients.shape)


# In[34]:


# print(minutes)
# print(tags)
# print(nutrition)
# print(steps)
# print(ingredients)


# In[39]:


# X contains minutes to cook, tags, nutritional values, cooking steps, ingredients
# X = [minutes, tags, nutrition, steps, ingredients]

# svd = TruncatedSVD(n_components=100, n_iter=10)
# svd.fit(X)
# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())








# ## 1. Individual Recommendations
# kNN, Item-based collaborative filtering

# In[90]:


users_sample = df_users.sample(n=1000)
recipes_sample = df_recipes.sample(n=1000)
print(users_sample)
print(recipes_sample)





#USE ONLY HEALTHY RECIPES

#recipe_ids = users_sample['item']
#healthy_recipes = ApplyHealthFilter(recipe_ids, [.15, .35, .10, .25])
#print(healthy_recipes)

#recipes_sample = df_recipes.copy()
#for i in range(len(df_recipes)):
#    for j in range(len(healthy_recipes)):
#        isHealthy = False
#        print(df_recipes.at[i, 'id'])
#        if df_recipes.at[i, 'id'] == healthy_recipes[j]:
#            isHealthy = True
#            break
#
#        if isHealthy != True:
#            recipes_sample.drop[i]
#recipes_sample





df_recipes = df_recipes.rename(columns={'id': 'recipe_id'})
df_users = df_users.rename(columns={'item': 'recipe_id'})
df = pd.merge(df_users, df_recipes, on='recipe_id')
df_sample = df.sample(n=1000)
print(df_sample)





#X = users_sample.drop('ratings', axis=1) # training data
#X = recipes_sample['calorie_level', 'techniques'] # training data

X = df_sample.iloc[:,[3,4]]; # training data HOW TO CONSIDER OTHER FEATURES ?
y = df_sample['rating']; # target values, labels

print(X)
print(y)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # 67% training, 33% testing

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train) # y_pred_train predicts X_train
y_pred_test = knn.predict(X_test)

#y_pred_train = np.rint(y_pred_train)
#y_pred_test = np.rint(y_pred_test)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)
# why float value ??
#print(y_pred_train)
#print(y_pred_test)





# TODO compare expected outputs with actual outputs

# TODO Solve
print('Accuracy on training data =', metrics.accuracy_score(np.array(y_train.to_list()), y_pred_train))
print('Accuracy on testing data =', metrics.accuracy_score(np.array(y_test.to_list()), y_pred_test))
print('')
print(metrics.classification_report(np.array(y_test.to_list()), y_pred_test))


#




df_recipes_full = pd.read_csv('Data/RAW_recipes.csv')
#Note that there are no duplicates
df_recipes_full.head()





df_recipes_full.drop('minutes', axis=1, inplace=True)
df_recipes_full.drop('contributor_id', axis=1, inplace=True)
df_recipes_full.drop('submitted', axis=1, inplace=True)
df_recipes_full.drop('tags', axis=1, inplace=True)
df_recipes_full.drop('nutrition', axis=1, inplace=True)
df_recipes_full.drop('steps', axis=1, inplace=True)
df_recipes_full.drop('ingredients', axis=1, inplace=True)
df_recipes_full.drop('description', axis=1, inplace=True)

df_recipes_full = df_recipes_full.rename(columns={'id': 'item'})

df_recipes_full.head()





users_ratings = df_users.groupby(['user']).count()  # count the ratings for each user
selected = users_ratings['rating'] > 30  # keep only 30 + ratings
selected_users = users_ratings.loc[selected]
random_selected = selected_users.sample(n=10)

select_column_df = random_selected.reset_index()[
    'user']  # reset_index() create a new index, and the userId became a column. Then, we can filter using the column name
group_users = list(
    select_column_df)  # iloc select by index, since our dataframe only has one row we read it from the index 0
print(group_users)





group_ratings = df_users.loc[df_users['user'].isin(group_users)]
total_recipes = set(df_recipes.index.tolist())
num_ratings_df = df_users.groupby(['item']).count()
considered_recipes = set(num_ratings_df.loc[num_ratings_df['user'] >= 30].reset_index()['item'])

group_seen_recipes = set(group_ratings['item'].tolist())
group_unseen_recipes = considered_recipes - group_seen_recipes

print('Total amount of recipes,', len(total_recipes))
print('Recipes that have at least 20 ratings,', len(considered_recipes))
print('Recipes that have been rated by the currently selected group,', len(group_seen_recipes))
print('New recipes that the group didnt try yet,', len(group_unseen_recipes))





from IPython.core.display import display
import itertools
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

user_user = UserUser(15, min_nbrs=3)  # Minimum (3) and maximum (12) number of neighbors to consider
recsys = Recommender.adapt(user_user)
recsys.fit(df_users)
group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
group_unseen_df = group_unseen_df[
    group_unseen_df['predicted_rating'].notna()]  # remove the recipes we couldn't get a prediction for
display(group_unseen_df.head(10))





#Min-Max normalization of predicted_ratings

maxVal = group_unseen_df['predicted_rating'].max()
minVal = group_unseen_df['predicted_rating'].min()
group_unseen_df['predicted_rating'] = (group_unseen_df['predicted_rating'] - minVal) / (
        maxVal - minVal)  # Normalized to 0 - 1 scale
group_unseen_df['predicted_rating'] *= 5  # Normalized to 0 - 5 scale

display(group_unseen_df.head(10))


# #### Least Misery strategy




least_misery_df = group_unseen_df.groupby(['item']).min().reset_index()
# TODO: Find name of recipe from the RAW data
least_misery_df = least_misery_df.join(df_recipes_full['name'], on='item')
items_lm = least_misery_df['item'].copy()
healthy_lm = ApplyHealthFilter(items_lm)

least_misery_df['healthy'] = least_misery_df['item'].apply(lambda x: 1 if x in healthy_lm else 0)
least_misery_df = least_misery_df.sort_values(by="predicted_rating", ascending=False)[
    ['item', 'predicted_rating', 'name', 'healthy']]

least_misery_df = least_misery_df[least_misery_df.healthy == 1]
display(least_misery_df.head(10))


# #### Most Pleasure strategy




most_pleasure_df = group_unseen_df.groupby(['item']).max().reset_index()
# TODO: Find name of recipe from the RAW data
items_mp = most_pleasure_df['item'].copy()
healthy_mp = ApplyHealthFilter(items_mp)

most_pleasure_df['healthy'] = most_pleasure_df['item'].apply(lambda x: 1 if x in healthy_mp else 0)

most_pleasure_df = most_pleasure_df.join(df_recipes_full['name'], on='item').reset_index()
most_pleasure_df = most_pleasure_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
    ['item', 'predicted_rating', 'name', 'healthy']]

most_pleasure_df = most_pleasure_df[most_pleasure_df.healthy == 1]

display(most_pleasure_df.head(10))


# #### Approval Voting




group_unseen_temp_df = group_unseen_df.copy()
group_unseen_temp_df['voted'] = group_unseen_temp_df['predicted_rating'].apply(lambda x: 1 if x > 3.5 else 0)
approval_df = group_unseen_temp_df.groupby(['item']).sum()
approval_df.drop('user', axis=1, inplace=True)
approval_df['predicted_rating'] /= len(random_selected)  # Normalize rating
# Only keep the items with maximum approval
approval_df = approval_df[approval_df.voted == approval_df.voted.max()]
approval_df = approval_df.sort_values(by="predicted_rating",
                                      ascending=False).reset_index()  # Get the best rated items with max approval
approval_df = approval_df.join(df_recipes_full['name'], on='item')
display(approval_df.head(10))

