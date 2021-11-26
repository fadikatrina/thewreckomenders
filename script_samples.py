import pandas as pd
from logger import l

def generate_sample_csvs():
	l.info("Started generating samples of the data sets")
	nutrition_data = pd.read_csv('Data/nutrition.csv', sep=';', nrows=10)
	nutrition_data.to_csv(r'Data/samples/nutrition.csv', index=None, sep=';')
	recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',', nrows=10)
	recipe_data.to_csv(r'Data/samples/RAW_recipes.csv', index=None, sep=';')
	df_users = pd.read_csv('Data/processed/knn_users.csv', nrows=10)
	df_users.to_csv(r'Data/samples/knn_users.csv', index=None, sep=';')
	df_recipes_full = pd.read_csv('Data/processed/group_recipes.csv', nrows=10)
	df_recipes_full.to_csv(r'Data/samples/group_recipes.csv', index=None, sep=';')
	df_recipes = pd.read_csv('Data/processed/knn_recipes.csv', nrows=10)
	df_recipes.to_csv(r'Data/samples/knn_recipes.csv', index=None, sep=';')

	PP_users = pd.read_csv('Data/PP_users.csv', nrows=10)
	PP_users.to_csv(r'Data/samples/PP_users.csv', index=None, sep=';')
	PP_recipes = pd.read_csv('Data/PP_recipes.csv', nrows=10)
	PP_recipes.to_csv(r'Data/samples/PP_recipes.csv', index=None, sep=';')
	RAW_interactions = pd.read_csv('Data/RAW_interactions.csv', sep=',', nrows=10)
	RAW_interactions.to_csv(r'Data/samples/RAW_interactions.csv', index=None, sep=';')
	interactions_train = pd.read_csv('Data/interactions_train.csv', sep=',', nrows=10)
	interactions_train.to_csv(r'Data/samples/interactions_train.csv', index=None, sep=';')
	interactions_test = pd.read_csv('Data/interactions_test.csv', sep=',', nrows=10)
	interactions_test.to_csv(r'Data/samples/interactions_test.csv', index=None, sep=';')


if __name__ == "__main__":
	generate_sample_csvs()
