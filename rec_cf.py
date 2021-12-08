import itertools

from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
import pandas as pd
import helpers_data_load as hl
import expl_cf as expl


def user_user_cf_group(df_users, group_users, group_unseen_recipes):
    user_user = UserUser(12, min_nbrs=5)
    recsys = Recommender.adapt(user_user)
    recsys.fit(df_users)
    group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
    group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
    # remove the recipes we couldn't get a prediction for
    # enh:easy log how many recipes dont have metrics
    group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]
    group_unseen_df['predicted_rating'] = min_max_scaling(group_unseen_df['predicted_rating'])
    return group_unseen_df


def item_item_cf_group(df_users, group_users, group_unseen_recipes):
    item_item = ItemItem(12, min_nbrs=5)
    recsys = Recommender.adapt(item_item)
    recsys.fit(df_users)
    group_unseen_df = pd.DataFrame(list(itertools.product(group_users, group_unseen_recipes)), columns=['user', 'item'])
    group_unseen_df['predicted_rating'] = recsys.predict(group_unseen_df)
    # remove the recipes we couldn't get a prediction for
    # enh:easy log how many recipes dont have metrics
    group_unseen_df = group_unseen_df[group_unseen_df['predicted_rating'].notna()]
    group_unseen_df['predicted_rating'] = min_max_scaling(group_unseen_df['predicted_rating'])
    return group_unseen_df


def user_user_cf(df_users, df_recipes_full, num_recs, user_id):
    user_user = UserUser(12, min_nbrs=5)
    recsys = Recommender.adapt(user_user)
    recsys.fit(df_users)
    recommended_items = recsys.recommend(user_id, num_recs)
    recommended_items = recommended_items.join(df_recipes_full['name'], on='item')
    recommended_items['score'] = min_max_scaling(recommended_items['score'])
    row = recommended_items.iloc[0]
    expl.individual_user(row['name'], str(row['score']))


def item_item_cf(df_users, df_recipes_full, num_recs, user_id):
    item_item = ItemItem(12, min_nbrs=5)
    recsys = Recommender.adapt(item_item)
    recsys.fit(df_users)
    recommended_items = recsys.recommend(user_id, num_recs)
    recommended_items = recommended_items.join(df_recipes_full['name'], on='item')
    recommended_items['score'] = min_max_scaling(recommended_items['score'])

    row = recommended_items.iloc[0]
    expl.individual_item(row['name'], str(row['score']))


def min_max_scaling(column):
    max_val = column.max()
    min_val = column.min()
    # Normalize to 1-5 range
    column = (5 - 1) * (column - min_val) / (max_val - min_val) + 1

    return column



