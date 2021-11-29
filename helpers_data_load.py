import pandas as pd
from logger import l
import spacy

l.debug("Data load start")

# Non_personalised
df_popular = pd.read_csv('Data/processed/popular.csv')

# kNN
nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';')
recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')
df_users = pd.read_csv('Data/processed/knn_users.csv')

# Group
df_recipes_full = pd.read_csv('Data/processed/group_recipes.csv')

# NLP Spacy
nlp = spacy.load('en_core_web_sm')


l.debug("Data load finish")
