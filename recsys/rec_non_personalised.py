from utils.helpers_data_load import df_popular, pd
from expl.expl_non_personalised import explain_non_personalised
from utils.logger import l


def popularity_based():
	pd.set_option('display.max_columns', None)
	l.critical("Non personalised - popularity based recommendation:")
	l.critical(get_most_popular_recipes())
	explain_non_personalised()


def get_most_popular_recipes(count=10):
	return df_popular.head(count)

