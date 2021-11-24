import pandas as pd
from filter_healthy import apply_health_filter
import itertools
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from logger import l


# todo add explanations
# todo add evaluation
def group_recommender(df_users, df_recipes, df_recipes_full, chosen_strategy):
	"""
	:param chosen_strategy: 1 for most pleasure, 2 for approval voting, 3 for least misery
	"""

	if chosen_strategy not in [1, 2, 3]:
		raise ValueError("Group recommender: chosen strategy has to be 1 or 2 or 3")

	# todo switch this to the user provided parameters from main
	users_ratings = df_users.groupby(['user']).count()  # count the ratings for each user
	selected = users_ratings['rating'] > 30  # keep only 30 + ratings
	selected_users = users_ratings.loc[selected]
	random_selected = selected_users.sample(n=10)

	# reset_index() create a new index, and the userId became a column. Then, we can filter using the column name
	select_column_df = random_selected.reset_index()['user']
	# iloc select by index, since our dataframe only has one row we read it from the index 0
	group_users = list(select_column_df)

	group_ratings = df_users.loc[df_users['user'].isin(group_users)]
	total_recipes = set(df_recipes.index.tolist())
	num_ratings_df = df_users.groupby(['item']).count()
	considered_recipes = set(num_ratings_df.loc[num_ratings_df['user'] >= 30].reset_index()['item'])

	group_seen_recipes = set(group_ratings['item'].tolist())
	group_unseen_recipes = considered_recipes - group_seen_recipes

	l.info(f'Total amount of recipes: {len(total_recipes)}')
	l.info(f'Recipes that have at least 20 ratings, {len(considered_recipes)}')
	l.info(f'Recipes that have been rated by the currently selected group: {len(group_seen_recipes)}')
	l.info(f'New recipes that the group didnt try yet: {len(group_unseen_recipes)}')

	# exp would be interesting to try different algorithms for prediction and see their affect on the number of
	#  missing ratings and coverage/accuracy metrics
	user_user = UserUser(15, min_nbrs=3)  # Minimum (3) and maximum (12) number of neighbors to consider
	recsys = Recommender.adapt(user_user)
	recsys.fit(df_users)
	group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
	group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
	# remove the recipes we couldn't get a prediction for
	# enh log how many recipes dont have metrics
	group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]

	# Min-Max normalization of predicted_ratings
	max_val = group_unseen_df['predicted_rating'].max()
	min_val = group_unseen_df['predicted_rating'].min()
	# Normalized to 0 - 1 scale
	group_unseen_df['predicted_rating'] = (group_unseen_df['predicted_rating'] - min_val) / (max_val - min_val)
	# Normalized to 0 - 5 scale
	group_unseen_df['predicted_rating'] *= 5

	if chosen_strategy is 1:
		return strategy_most_pleasure(group_unseen_df, df_recipes_full)
	elif chosen_strategy is 2:
		return strategy_approval_voting(group_unseen_df, df_recipes_full)
	else:
		return strategy_least_misery(group_unseen_df, df_recipes_full, random_selected)


def strategy_least_misery(group_unseen_df, df_recipes_full):
	least_misery_df = group_unseen_df.groupby(['item']).min().reset_index()
	# TODO: Find name of recipe from the RAW data
	least_misery_df = least_misery_df.join(df_recipes_full['name'], on='item')
	items_lm = least_misery_df['item'].copy()
	healthy_lm = apply_health_filter(items_lm)

	least_misery_df['healthy'] = least_misery_df['item'].apply(lambda x: 1 if x in healthy_lm else 0)
	least_misery_df = least_misery_df.sort_values(by="predicted_rating", ascending=False)[['item', 'predicted_rating', 'name', 'healthy']]

	least_misery_df = least_misery_df[least_misery_df.healthy == 1]
	return least_misery_df


def strategy_most_pleasure(group_unseen_df, df_recipes_full):
	most_pleasure_df = group_unseen_df.groupby(['item']).max().reset_index()
	# TODO: Find name of recipe from the RAW data
	items_mp = most_pleasure_df['item'].copy()
	healthy_mp = apply_health_filter(items_mp)

	most_pleasure_df['healthy'] = most_pleasure_df['item'].apply(lambda x: 1 if x in healthy_mp else 0)

	most_pleasure_df = most_pleasure_df.join(df_recipes_full['name'], on='item').reset_index()
	most_pleasure_df = most_pleasure_df.sort_values(by="predicted_rating", ascending=False).reset_index()[['item', 'predicted_rating', 'name', 'healthy']]

	most_pleasure_df = most_pleasure_df[most_pleasure_df.healthy == 1]
	return most_pleasure_df


def strategy_approval_voting(group_unseen_df, df_recipes_full, random_selected):
	group_unseen_temp_df = group_unseen_df.copy()
	group_unseen_temp_df['voted'] = group_unseen_temp_df['predicted_rating'].apply(lambda x: 1 if x > 3.5 else 0)
	approval_df = group_unseen_temp_df.groupby(['item']).sum()
	approval_df.drop('user', axis=1, inplace=True)
	approval_df['predicted_rating'] /= len(random_selected)  # Normalize rating
	# Only keep the items with maximum approval
	approval_df = approval_df[approval_df.voted == approval_df.voted.max()]
	approval_df = approval_df.sort_values(by="predicted_rating", ascending=False).reset_index()  # Get the best rated items with max approval
	approval_df = approval_df.join(df_recipes_full['name'], on='item')
	return approval_df
