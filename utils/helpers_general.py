from helpers_data_load import df_users


def how_many_reviews(user_id):
	return len(get_user_reviews(user_id).index)


def get_user_reviews(user_id):
	return df_users.loc[df_users['user'] == user_id]


def sample_users(count):
	return df_users.sample(count)['user'].tolist()


# Returns how many users have less than the number of specified reviews
def how_many_users_have_few_reviews(reviews_number):
	number_of_users_few = df_users[df_users["n_ratings"] < reviews_number].shape[0]
	number_of_users_total = len(df_users.index)
	return number_of_users_few, number_of_users_total
