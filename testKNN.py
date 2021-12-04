from random import randrange

from rec_individual_knn import knn

df_recipes = None
import pandas as pd


if __name__ == "__main__":
	recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')
	df_users = pd.read_csv('Data/processed/knn_users.csv')
	rand = randrange(len(df_users))
	knn(df_users, recipe_data, rand, False)
