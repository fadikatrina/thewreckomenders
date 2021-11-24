import pandas as pd
from ast import literal_eval
from logger import l
from pathlib import Path
import os


if __name__ == "__main__":
	l.info("Data prep start")

	# Directory management
	Path("/Data/processed").mkdir(parents=True, exist_ok=True)
	if os.path.isfile("Data/processed/knn_recipes.csv"):
		input("Seems like data prep is already done, no need to run this step everytime, if you insist press enter to continue")
	try:
		os.remove("Data/processed/knn_recipes.csv")
		os.remove("Data/processed/knn_users.csv")
		os.remove("Data/processed/group_recipes.csv")
	except OSError:
		l.debug("Could not delete one or more file, maybe they dont exist anyways")
		pass

	# individual recommenders preparation - kNN
	df_users = pd.read_csv('Data/PP_users.csv')
	df_recipes = pd.read_csv('Data/PP_recipes.csv')

	df_users.drop('techniques', axis=1, inplace=True)
	df_users.drop('n_items', axis=1, inplace=True)
	df_users.drop('n_ratings', axis=1, inplace=True)
	df_users = df_users.rename(columns={'u': 'user', 'items': 'item', 'ratings': 'rating'})
	df_users['rating'] = df_users['rating'].apply(literal_eval)
	df_users['item'] = df_users['item'].apply(literal_eval)
	df_users = df_users.explode(['rating', 'item'], ignore_index=True)
	df_users.to_csv("Data/processed/knn_users.csv")

	df_recipes.drop('i', axis=1, inplace=True)
	df_recipes.drop('name_tokens', axis=1, inplace=True)
	df_recipes.drop('ingredient_tokens', axis=1, inplace=True)
	df_recipes.drop('steps_tokens', axis=1, inplace=True)
	df_recipes['techniques'] = df_recipes['techniques'].apply(literal_eval)
	df_recipes['ingredient_ids'] = df_recipes['ingredient_ids'].apply(literal_eval)
	df_recipes = df_recipes.explode('techniques')
	df_recipes = df_recipes.explode('ingredient_ids')
	df_recipes.to_csv("Data/processed/knn_recipes.csv")

	# group recommenders preparation
	df_recipes_full = pd.read_csv('Data/RAW_recipes.csv')
	df_recipes_full.drop('minutes', axis=1, inplace=True)
	df_recipes_full.drop('contributor_id', axis=1, inplace=True)
	df_recipes_full.drop('submitted', axis=1, inplace=True)
	df_recipes_full.drop('tags', axis=1, inplace=True)
	df_recipes_full.drop('nutrition', axis=1, inplace=True)
	df_recipes_full.drop('steps', axis=1, inplace=True)
	df_recipes_full.drop('ingredients', axis=1, inplace=True)
	df_recipes_full.drop('description', axis=1, inplace=True)
	df_recipes_full = df_recipes_full.rename(columns={'id': 'item'})
	df_recipes_full.to_csv("Data/processed/group_recipes.csv")

	l.info("Data prep finish")
