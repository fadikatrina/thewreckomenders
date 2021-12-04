import pandas as pd
from IPython.core.display import display

from filter_healthy import apply_health_filter
from rec_cf import user_user_cf_group
from rec_cf import item_item_cf_group
from logger import l
from helpers_data_load import df_users, df_recipes_full
import expl_cf

df_recipes = None


# todo:high add explanations
# todo:high add evaluation
def group_recommender(df_users_gr, df_recipes_full_gr, chosen_strategy, chosen_useritem):
    """
    :param chosen_useritem: 1 for UserUser, 2 for ItemItem
    :param chosen_strategy: 1 for most pleasure, 2 for approval voting, 3 for least misery
    """
    if chosen_strategy not in [1, 2, 3]:
        raise ValueError("Group recommender: chosen strategy has to be 1 or 2 or 3")

    if chosen_useritem not in [1, 2]:
        raise ValueError("Group recommender: chosen algorithm has to be 1 or 2")

    # todo:high switch this to the user provided parameters from main
    users_ratings = df_users_gr.groupby(['user']).count()  # count the ratings for each user
    selected = users_ratings['rating'] > 30  # keep only 30 + ratings
    selected_users = users_ratings.loc[selected]
    random_selected = selected_users.sample(n=10)

    # reset_index() create a new index, and the userId became a column. Then, we can filter using the column name
    select_column_df = random_selected.reset_index()['user']
    # iloc select by index, since our dataframe only has one row we read it from the index 0
    group_users = list(select_column_df)

    group_ratings = df_users_gr.loc[df_users_gr['user'].isin(group_users)]
    total_recipes = set(df_recipes_full_gr.index.tolist())
    num_ratings_df = df_users_gr.groupby(['item']).count()
    considered_recipes = set(num_ratings_df.loc[num_ratings_df['user'] >= 30].reset_index()['item'])

    group_seen_recipes = set(group_ratings['item'].tolist())
    group_unseen_recipes = considered_recipes - group_seen_recipes

    l.info(f'Total amount of recipes: {len(total_recipes)}')
    l.info(f'Recipes that have at least 30 ratings, {len(considered_recipes)}')
    l.info(f'Recipes that have been rated by the currently selected group: {len(group_seen_recipes)}')
    l.info(f'New recipes that the group didnt try yet: {len(group_unseen_recipes)}')

    # exp:easy would be interesting to try different algorithms for prediction and see their affect on the number of
    #  missing ratings and coverage/accuracy metrics

    # todo:med switch these variables from being provided to CF here to being found by the CF itself,
    #  i.e. better decoupling
    if chosen_useritem == 1:
        group_unseen_df = user_user_cf_group(df_users_gr, group_users, group_unseen_recipes)
    else:
        group_unseen_df = item_item_cf_group(df_users_gr, group_users, group_unseen_recipes)

    # Min-Max normalization of predicted_ratings
    max_val = group_unseen_df['predicted_rating'].max()
    min_val = group_unseen_df['predicted_rating'].min()
    # Normalized to 0 - 1 scale
    group_unseen_df['predicted_rating'] = (group_unseen_df['predicted_rating'] - min_val) / (max_val - min_val)
    # Normalized to 0 - 5 scale
    group_unseen_df['predicted_rating'] *= 5

    if chosen_strategy == 1:
        return strategy_most_pleasure(group_unseen_df, df_recipes_full_gr)
    elif chosen_strategy == 2:
        return strategy_approval_voting(group_unseen_df, df_recipes_full_gr, random_selected)
    else:
        return strategy_least_misery(group_unseen_df, df_recipes_full_gr)


def strategy_least_misery(group_unseen_df, df_recipes_full_gr):
    least_misery_df = group_unseen_df.groupby(['item']).min().reset_index()
    least_misery_df = least_misery_df.join(df_recipes_full_gr['name'], on='item')
    items_lm = least_misery_df['item'].copy()
    unhealthy_choice = least_misery_df['name'].iloc[0]
    healthy_lm = apply_health_filter(items_lm)

    least_misery_df['healthy'] = least_misery_df['item'].apply(lambda x: 1 if x in healthy_lm else 0)
    least_misery_df = least_misery_df.sort_values(by="predicted_rating", ascending=False)[
        ['item', 'predicted_rating', 'name', 'healthy']]

    least_misery_df = least_misery_df[least_misery_df.healthy == 1]
    expl_cf.expl_least_misery(least_misery_df['name'].iloc[0], unhealthy_choice,
                              str(least_misery_df['predicted_rating'].iloc[0]))

    return least_misery_df


def strategy_most_pleasure(group_unseen_df, df_recipes_full_gr):
    most_pleasure_df = group_unseen_df.groupby(['item']).max().reset_index()
    most_pleasure_df = most_pleasure_df.join(df_recipes_full_gr['name'], on='item')
    items_mp = most_pleasure_df['item'].copy()
    unhealthy_choice = most_pleasure_df['name'].iloc[0]
    healthy_mp = apply_health_filter(items_mp)

    most_pleasure_df['healthy'] = most_pleasure_df['item'].apply(lambda x: 1 if x in healthy_mp else 0)
    most_pleasure_df = most_pleasure_df.sort_values(by="predicted_rating", ascending=False)[
        ['item', 'predicted_rating', 'name', 'healthy']]

    most_pleasure_df = most_pleasure_df[most_pleasure_df.healthy == 1]
    expl_cf.expl_most_pleasure(most_pleasure_df['name'].iloc[0], unhealthy_choice,
                               str(most_pleasure_df['predicted_rating'].iloc[0]))

    return most_pleasure_df


def strategy_approval_voting(group_unseen_df, df_recipes_full_gr, random_selected):
    group_unseen_temp_df = group_unseen_df.copy()
    group_unseen_temp_df['voted'] = group_unseen_temp_df['predicted_rating'].apply(lambda x: 1 if x > 3.5 else 0)
    approval_df = group_unseen_temp_df.groupby(['item']).sum()
    approval_df.drop('user', axis=1, inplace=True)
    approval_df['predicted_rating'] /= len(random_selected)  # Normalize rating
    # Only keep the items with maximum approval
    approval_df = approval_df[approval_df.voted == approval_df.voted.max()]
    approval_df = approval_df.sort_values(by="predicted_rating",
                                          ascending=False).reset_index()  # Get the best rated items with max approval
    approval_df = approval_df.join(df_recipes_full_gr['name'], on='item')
    expl_cf.expl_approval_voting(approval_df['name'].iloc[0], str(approval_df['voted'].iloc[0]),
                                 str(approval_df['predicted_rating'].iloc[0]))

    return approval_df


group_recommender(df_users, df_recipes_full, 1, 1)
group_recommender(df_users, df_recipes_full, 3, 2)
group_recommender(df_users, df_recipes_full, 2, 1)
