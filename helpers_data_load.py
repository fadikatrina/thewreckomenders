import pandas as pd
from logger import l

l.debug("Data load start")
# df_recipes is loaded in kNN, and group recommender

# kNN
nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';')
recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')
df_users = pd.read_csv('Data/processed/knn_users.csv')

# Group
df_recipes_full = pd.read_csv('Data/processed/group_recipes.csv')

l.debug("Data load finish")

