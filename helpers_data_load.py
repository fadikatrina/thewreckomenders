import pandas as pd
from logger import l
import spacy

l.debug("Data load start")

df_popular = pd.read_csv('Data/processed/popular.csv')
nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';')
df_users = pd.read_csv('Data/processed/knn_users.csv')
df_users['item'] = df_users['item'].astype('Int64')
df_recipes_full = pd.read_csv('Data/processed/group_recipes.csv')
# nlp = spacy.load('en_core_web_sm')

recipes_raw = None
interactions_raw = None


def filter_data(config):
	global df_users
	global df_recipes_full
	global recipes_raw

	l.debug(f"Secondary data load with filtering started, config {config}")
	load_recipes_raw()
	load_interactions_raw()

	l.debug("Secondary data load with filtering finished")


def load_recipes_raw():
	global recipes_raw
	l.debug("Loading RAW_Recipes")
	recipes_raw = pd.read_csv('Data/RAW_recipes.csv')


def load_interactions_raw():
	global interactions_raw
	l.debug("Loading RAW_Interactions")
	interactions_raw = pd.read_csv('Data/RAW_interactions.csv')


l.debug("Initial data load finish")
