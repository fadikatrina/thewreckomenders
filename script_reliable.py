import pandas as pd
from helpers_data_load import df_users


def generate_reliable_csv():
	df_sum_of_ratings = df_users.groupby("item").agg(
		sum_of_ratings=pd.NamedAgg(column="rating", aggfunc=sum)
	)
	df_recipes = pd.read_csv('Data/RAW_recipes.csv')
	df_sum_of_ratings_details = pd.merge(df_sum_of_ratings, df_recipes, how='inner', left_on='item', right_on='id')
	df_sum_of_ratings_user = df_sum_of_ratings_details.groupby("contributor_id").agg(
		sum_of_ratings_user=pd.NamedAgg(column="sum_of_ratings", aggfunc=sum)
	)
	df_sum_of_ratings_user.to_csv("Data/processed/reliable.csv")


if __name__ == "__main__":
	generate_reliable_csv()
