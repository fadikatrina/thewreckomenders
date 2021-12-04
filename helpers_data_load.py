import pandas as pd
from logger import l
import spacy

l.debug("Data load start")

# Non_personalised
df_popular = pd.read_csv('Data/processed/popular.csv')

# kNN
nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';')
df_users = pd.read_csv('Data/processed/knn_users.csv')
df_users['item'] = df_users['item'].astype('Int64')

# Group
df_recipes_full = pd.read_csv('Data/processed/group_recipes.csv')

# NLP Spacy
# nlp = spacy.load('en_core_web_sm')

recipes_raw = None
interactions_raw = None
knn_recipes = None


def load_recipes_raw():
	global recipes_raw
	l.debug("Loading RAW_Recipes")
	recipes_raw = pd.read_csv('Data/RAW_recipes.csv')


def load_interactions_raw():
	global interactions_raw
	l.debug("Loading RAW_Interactions")
	interactions_raw = pd.read_csv('Data/RAW_interactions.csv')


def load_knn_recipes():
	global knn_recipes
	l.debug("Loading knn_recipes")
	knn_recipes = pd.read_csv('Data/processed/knn_recipes.csv')


l.debug("Data load finish")
