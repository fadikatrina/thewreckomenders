import pandas as pd
from utils.logger import l


def generate_popular_csv():
	l.info("Started generating csv with popular recipes")
	df_interactions = pd.read_csv('../Data/RAW_interactions.csv')
	df_recipes = pd.read_csv('../Data/RAW_recipes.csv')
	df_recipes.drop(
		['contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients'],
		 axis=1, inplace=True)

	df_sum_of_ratings = df_interactions.groupby("recipe_id").agg(
		sum_of_ratings=pd.NamedAgg(column="rating", aggfunc=sum)
	)
	df_sum_of_ratings = df_sum_of_ratings.sort_values('sum_of_ratings', ascending=False)
	df_sum_of_ratings = df_sum_of_ratings.head(100)
	df_ratings_details = pd.merge(df_sum_of_ratings, df_recipes, how='inner', left_on='recipe_id', right_on='id')
	df_ratings_details.to_csv("Data/processed/popular.csv")


if __name__ == "__main__":
	generate_popular_csv()
