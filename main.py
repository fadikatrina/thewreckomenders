from logger import l
from helpers import sample_users, how_many_reviews
from non_personalised import popularity_based
from content_based import tf_idf
from individual import knn
from group import group_recommender
from data_load import df_users, df_recipes, recipe_data, df_recipes_full


settings = {
	"recommender_type":0,
	"min_reviews_for_personalised":5,
	"min_recipe":20,
	"min_user":5,
	"max_duration":100,
	"max_steps":5,
	"filter_healthy":0,
	"filter_healthy_post":1,
	"user_specify_users":0,
	"users":[1, 2],
	"more_recs":False,
	"relationship":1,
	"keywords":["mexican", "spicy"],
}


# enh:med make one function that all the algorithms call with the recommendations and explanations so the output is
#  comparable/measurable easily
# todo actually filter according to the variables input by the user
def start(config={}):
	l.debug("Application start")
	if not config:
		config["recommender_type"] = int(input("Input 0 for INDIVIDUAL recommender, and 1 for GROUP recommender ") or "0")
		l.info(f"recommender_type {config['recommender_type']}")
		config["min_reviews_for_personalised"] = int(input("Input min # of reviews a user should have for personalised rec ") or "5")
		l.info(f"min_reviews_for_personalised {config['min_reviews_for_personalised']}")
		config["min_recipe"] = int(input("Input the MIN number of reviews a RECIPE should have ") or "20")
		l.info(f"min_recipe {config['min_recipe']}")
		config["min_user"] = int(input("Input the MIN number of reviews a USER should have ") or "5")
		l.info(f"min_user {config['min_user']}")
		config["max_duration"] = int(input("Input the MAX DURATION a recipe should take to make ") or "100")
		l.info(f"max_duration {config['max_duration']}")
		config["max_steps"] = int(input("Input the MAX STEPS a recipe should take to make ") or "5")
		l.info(f"max_steps {config['max_steps']}")
		config["filter_healthy"] = int(input("Input 1 to turn the healthy filter on, 0 for off ") or "0")
		l.info(f"filter_healthy {config['filter_healthy']}")
		config["filter_healthy_post"] = 1
		if config['filter_healthy']:
			config["filter_healthy_post"] = int(input("Input 0 for PRE healthy filtering, 1 for POST ") or "1")
		l.info(f"filter_healthy_post {config['filter_healthy_post']}")
		config["user_specify_users"] = int(input("Input 0 to select user(s) AUTOMATICALLY, 1 to specify MANUALLY") or "0")
		l.info(f"user_specify_users {config['user_specify_users']}")
		config["users"] = []
		if config['user_specify_users']:
			users = [int(x) for x in input("Specify the ids of user(s) separated by spaces").split()]
		else:
			number_of_users = 1
			if config['recommender_type']:
				number_of_users = int(input("Input number of users to sample") or "20")
			users = sample_users(number_of_users)
		l.info(f"users {config['users']}")
	more_recs = True
	while more_recs:
		if recommender_type:
			l.debug("Group recommendations branch")
			relationship = int(input("What is the type of relationship between the users? ") or "1")
			l.info(f"relationship {relationship}")
			l.debug("Group recommender started")
			group_recommendation = group_recommender(df_users, df_recipes, df_recipes_full, relationship)
			l.info(f"group_recommendation {group_recommendation}")
		else:
			l.debug("Individual recommendations branch")
			if len(users) > 1:
				raise ValueError("If individual recommendations is picked, a maximum of 1 user id can be specified")
			user_id = users["user"]
			user_reviews_count = how_many_reviews(user_id)
			if user_reviews_count < min_reviews_for_personalised:
				l.info(f"Because the user has less than {min_reviews_for_personalised} ratings, non-personalised popularity based recommendations will be made")
				popularity_based()
				# exp:easy evaluation of effect of different number of keywords
				keywords = [x for x in input("If you would like content-based recommendations based on keywords you specify, please specify the keywords seperated by spaces. Do not enter keywords to skip.").split()]
				if len(keywords) == 0:
					l.info("No keywords provided, exiting")
					exit(0)
				l.debug("Starting TF-IDF")
				# fixme data for tf_idf
				tf_idf(keywords)
			else:
				l.info("The user has more than 5 reviews, individual recommendations using kNN")
				individual_recommendation = knn(df_users, df_recipes, recipe_data)
				l.info(f"individual_recommendation {individual_recommendation}")
		more_recs = int(input("Input 0 to quit, 1 to try another algorithm") or "0")


# enh could parallelize the start part, as the data is loaded at the begging and not for each start
if __name__ == "__main__":
	"""
		start() without params prompts the user to select each variable at runtime, 
		if no variables are selected their are defaults
		start(settings) accepts a dict and uses that dict values instead of acting the user
		You can run multiple experiments very quickly by using more than one start with different settings after each other
	"""
	start(settings)
	# start()
