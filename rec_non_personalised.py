from helpers_data_load import df_popular, pd
from logger import l


def popularity_based():
	pd.set_option('display.max_columns', None)
	l.critical("Non personalised - popularity based recommendation:")
	l.critical(get_most_popular_recipes())
	l.critical("Unfortunately we do not know enough about your preferences to give a more personalised recommendation "
	           "but these are some recipes that are popular among other users, check them out! Do not forget to rate "
	           "the ones you like/dislike.")


def get_most_popular_recipes(count=10):
	return df_popular.head(count)

popularity_based()
