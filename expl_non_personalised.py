from logger import l


def explain_non_personalised():
	l.critical("Unfortunately we do not know enough about your preferences to give a more personalised recommendation "
	           "but these are some recipes that are popular among other users, check them out! Do not forget to rate "
	           "the ones you like/dislike.")
	if int(input("Input 1 to get more details or enter to move on ") or "0"):
		l.critical("In order to give you the most popular recipes, we take into account both the ratings other people "
		           "gave and how many users rated the item. The more users that rated it the higher it is.")
