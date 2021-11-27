from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import coverage_error
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from helpers_data_load import df_recipes_full, df_users


def hold_out_uu(test_size):
    train_df, test_df = train_test_split(df_users, test_size=test_size)

    user_user = UserUser(15, min_nbrs=3)
    recsys = Recommender.adapt(user_user)
    recsys.fit(train_df)

    test_df['predicted_rating'] = recsys.predict(test_df)

    test_df['relevant'] = test_df['rating'].apply(lambda x: 1 if x > 3 else 0)

    test_df['predicted_relevant'] = test_df['predicted_rating'].apply(lambda x: 1 if x > 3 else 0)

    y_test = list(test_df['relevant'])
    y_pred = list(test_df['predicted_relevant'])

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print('Precision:', precision, '\nRecall:', recall, '\nFscore:', fscore, '\nRMSE:', rmse)


def hold_out_ii(test_size):
    train_df, test_df = train_test_split(df_users, test_size=test_size)

    item_item = ItemItem(15, min_nbrs=3)
    recsys = Recommender.adapt(item_item)
    recsys.fit(train_df)

    test_df['predicted_rating'] = recsys.predict(test_df)

    test_df['relevant'] = test_df['rating'].apply(lambda x: 1 if x > 3 else 0)

    test_df['predicted_relevant'] = test_df['predicted_rating'].apply(lambda x: 1 if x > 3 else 0)

    y_test = list(test_df['relevant'])
    y_pred = list(test_df['predicted_relevant'])

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print('Precision:', precision, '\nRecall:', recall, '\nFscore:', fscore, '\nRMSE:', rmse)


hold_out_uu(0.1)
hold_out_ii(0.1)