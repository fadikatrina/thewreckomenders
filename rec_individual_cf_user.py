import itertools
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
import pandas as pd


def user_user_cf(df_users, group_users, group_unseen_recipes, group_flag=True):
	user_user = UserUser(15, min_nbrs=3)
	recsys = Recommender.adapt(user_user)
	recsys.fit(df_users)
	group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
	group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
	# remove the recipes we couldn't get a prediction for
	# enh:easy log how many recipes dont have metrics
	group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]
