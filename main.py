from logger import l
from helpers_general import sample_users, how_many_reviews
from rec_non_personalised import popularity_based
from rec_individual_content_based import fasttext_keywords
from rec_individual_knn import knn
from rec_group import group_recommender
import helpers_data_load as dl

settings = {
	"recommender_type": 1,
	"min_reviews_for_personalised": 5,
	"min_recipe": 20,
	"min_user": 5,
	"max_duration": 100,
	"max_steps": 5,
	"filter_healthy": 0,
	"filter_healthy_post": 1,
	"user_specify_users": 0,
	"number_of_users": 10,
	"users": [],
	"more_recs": False,
	"relationship": 1,
	"keywords": ["mexican", "spicy"],
}


# URGENT fix the way knn is called here
# URGENT change cut off between non-personalised and rest to 7
# URGENT change to cf for 15 and above reviews
# URGENT user picks between itemitem and useruser
# URGENT filter from data load based on these values
def start(config={}):
	l.debug("Application start")
	if not config:
		config["recommender_type"] = int(
			input("Input 0 for INDIVIDUAL recommender, and 1 for GROUP recommender ") or "0")
		config["min_reviews_for_personalised"] = int(
			input("Input min # of reviews a user should have for personalised rec ") or "5")
		config["min_recipe"] = int(input("Input the MIN number of reviews a RECIPE should have ") or "20")
		config["min_user"] = int(input("Input the MIN number of reviews a USER should have ") or "5")
		config["max_duration"] = int(input("Input the MAX DURATION a recipe should take to make ") or "100")
		config["max_steps"] = int(input("Input the MAX STEPS a recipe should take to make ") or "5")
		config["filter_healthy"] = int(input("Input 1 to turn the healthy filter on, 0 for off ") or "0")
		config["filter_healthy_post"] = 1
		if config['filter_healthy']:
			config["filter_healthy_post"] = int(input("Input 0 for PRE healthy filtering, 1 for POST ") or "1")
		config["user_specify_users"] = int(
			input("Input 0 to select user(s) AUTOMATICALLY, 1 to specify MANUALLY") or "0")
		config["users"] = []
		if config['user_specify_users']:
			config['users'] = [int(x) for x in input("Specify the ids of user(s) separated by spaces").split()]
	if not config["user_specify_users"]:
		if config['recommender_type']:
			if not "number_of_users" in config:
				config['number_of_users'] = int(input("Input number of users to sample") or "20")
		else:
			config['number_of_users'] = 1
		config['users'] = sample_users(config['number_of_users'])
	l.info(config)
	more_recs = True
	while more_recs:
		if config["recommender_type"]:
			l.debug("Group recommendations branch")
			if not "relationship" in config:
				config["relationship"] = int(input("What is the type of relationship between the users? ") or "1")
				l.info(f"relationship {config['relationship']}")
			l.debug("Group recommender started")
			group_recommendation = group_recommender(dl.df_users, dl.df_recipes_full, config["relationship"])
			l.info(f"group_recommendation {group_recommendation}")
		else:
			l.debug("Individual recommendations branch")
			if len(config['users']) > 1:
				raise ValueError("If individual recommendations is picked, a maximum of 1 user id can be specified")
			user_id = config['users'][0]
			user_reviews_count = how_many_reviews(user_id)
			if user_reviews_count < config["min_reviews_for_personalised"]:
				l.info(
					f"Because the user has less than {config['min_reviews_for_personalised']} ratings, non-personalised popularity based recommendations will be made")
				popularity_based()
				if not config["keywords"]:
					config['keywords'] = [x for x in input(
						"If you would like content-based recommendations based on keywords you specify, please specify the keywords seperated by spaces. Do not enter keywords to skip.").split()]
					l.info(f"keywords {config['keywords']}")
				if len(config['keywords']) == 0:
					l.info("No keywords provided, exiting")
					exit(0)
				l.debug("Starting Content Based Individual Recommedner")
				fasttext_keywords(config['keywords'])
			else:
				l.info(
					f"The user has {user_reviews_count}, which is more than {config['min_reviews_for_personalised']} reviews, making individual recommendations using kNN")
				individual_recommendation = knn()
				l.info(f"individual_recommendation {individual_recommendation}")
		more_recs = int(input("Input 0 to quit, 1 to try another algorithm") or "0")


# enh:med could parallelize the start function for more than 1 setting file,
#  as the data is loaded at the beginning and not for each start call
#  could be an issue with the way the big recipes df is loaded in knn and group
if __name__ == "__main__":
	"""
		start() without params prompts the user to select each variable at runtime, 
		if no variables are selected their are defaults
		start(settings) accepts a dict and uses that dict values instead of acting the user
		You can run multiple experiments very quickly by using more than one start with different settings after each other
	"""
	# start(settings)
	start()
