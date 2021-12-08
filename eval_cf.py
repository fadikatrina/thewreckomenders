from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from helpers_data_load import df_recipes_full, df_users


def hold_out_uu(test_size):
    train_df, test_df = train_test_split(df_users, test_size=test_size)

    user_user = UserUser(12, min_nbrs=5)
    recsys = Recommender.adapt(user_user)
    recsys.fit(train_df)

    test_df['predicted_rating'] = recsys.predict(test_df)
    full_length = len(test_df['predicted_rating'])
    test_df['relevant'] = test_df['rating'].apply(lambda x: 1 if x > 3 else 0)
    test_df['predicted_relevant'] = test_df['predicted_rating'].apply(lambda x: 1 if x > 3 else 0)
    y_test = list(test_df['relevant'])
    y_pred = list(test_df['predicted_relevant'])

    test_df = test_df[test_df['predicted_rating'].notna()]
    partial_length = len(test_df['predicted_rating'])

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    rmse = mean_squared_error(list(test_df['rating']), list(test_df['predicted_rating']), squared=False)
    coverage = partial_length / full_length

    print('---UserUser Metrics Results---')
    print('Hold-Out Evaluation; test_size =', test_size)
    print('Precision:', precision, '\nRecall:', recall, '\nFscore:', fscore, '\nRMSE:', rmse, '\nCoverage:', coverage)


def hold_out_ii(test_size):
    train_df, test_df = train_test_split(df_users, test_size=test_size)

    item_item = ItemItem(12, min_nbrs=5)
    recsys = Recommender.adapt(item_item)
    recsys.fit(train_df)

    test_df['predicted_rating'] = recsys.predict(test_df)
    full_length = len(test_df['predicted_rating'])
    test_df['relevant'] = test_df['rating'].apply(lambda x: 1 if x > 3 else 0)
    test_df['predicted_relevant'] = test_df['predicted_rating'].apply(lambda x: 1 if x > 3 else 0)
    y_test = list(test_df['relevant'])
    y_pred = list(test_df['predicted_relevant'])

    test_df = test_df[test_df['predicted_rating'].notna()]
    partial_length = len(test_df['predicted_rating'])

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    rmse = mean_squared_error(list(test_df['rating']), list(test_df['predicted_rating']), squared=False)
    coverage = partial_length / full_length

    print('---ItemItem Metrics Results---')
    print('Hold-Out Evaluation; test_size =', test_size)
    print('Precision:', precision, '\nRecall:', recall, '\nFscore:', fscore, '\nRMSE:', rmse, '\nCoverage:', coverage)


hold_out_uu(0.1)
hold_out_ii(0.1)
