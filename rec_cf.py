import itertools
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
import pandas as pd


def user_user_cf_group(df_users, group_users, group_unseen_recipes):
    user_user = UserUser(15, min_nbrs=3)
    recsys = Recommender.adapt(user_user)
    recsys.fit(df_users)
    group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
    group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
    # remove the recipes we couldn't get a prediction for
    # enh:easy log how many recipes dont have metrics
    group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]
    return group_unseen_df


def item_item_cf_group(df_users, group_users, group_unseen_recipes):
    item_item = ItemItem(15, min_nbrs=3)
    recsys = Recommender.adapt(item_item)
    recsys.fit(df_users)
    group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
    group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
    # remove the recipes we couldn't get a prediction for
    # enh:easy log how many recipes dont have metrics
    group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]
    return group_unseen_df


def user_user_cf(df_users, df_recipes_full, num_recs, user_id):
    user_user = UserUser(15, min_nbrs=3)
    recsys = Recommender.adapt(user_user)
    recsys.fit(df_users)
    recommended_items = recsys.recommend(user_id, num_recs)
    return recommended_items.join(df_recipes_full['name'], on='item')


def item_item_cf(df_users, df_recipes_full, num_recs, user_id):
    item_item = ItemItem(15, min_nbrs=3)
    recsys = Recommender.adapt(item_item)
    recsys.fit(df_users)
    recommended_items = recsys.recommend(user_id, num_recs)
    return recommended_items.join(df_recipes_full['name'], on='item')
