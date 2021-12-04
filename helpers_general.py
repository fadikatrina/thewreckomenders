import helpers_data_load as dl
from logger import l
import string

interactions_raw = None


def how_many_reviews(user_id):
	return len(get_user_reviews(user_id).index)


def get_user_reviews(user_id):
	return dl.df_users.loc[dl.df_users['user'] == user_id]


def sample_users(count):
	return dl.df_users.sample(count)['user'].tolist()


# Returns how many users have less than the number of specified reviews
def how_many_users_have_few_reviews(reviews_number):
	number_of_users_few = dl.df_users[dl.df_users["n_ratings"] < reviews_number].shape[0]
	number_of_users_total = len(dl.df_users.index)
	return number_of_users_few, number_of_users_total


# Given string, returns it tokenized, lemmatized, filtered (stop words), without punctuation
def clean_text(text, domain_stopwords=[]):
	try:
		doc = dl.nlp(text)
	except ValueError:
		l.warning(f"Spacy does not understand '{text}' retunring empty string")
		return ""

	# Tokenization and lemmatization are done with the spacy nlp pipeline commands
	lemma_list = []
	for token in doc:
		lemma_list.append(token.lemma_)

	# Filter the stopword
	filtered_sentence = []
	for word in lemma_list:
		lexeme = dl.nlp.vocab[word]
		if word not in domain_stopwords:
			if not lexeme.is_stop:
				filtered_sentence.append(word)

	filtered_sentence = " ".join(filtered_sentence)
	return filtered_sentence.translate(str.maketrans('', '', string.punctuation))


def get_recipe_details(recipe_id):
	if dl.recipes_raw is None:
		dl.load_recipes_raw()
	return dl.recipes_raw.loc[dl.recipes_raw['id'] == recipe_id]


def get_all_recipes_of_user(user_id):
	l.info(f"Getting all the recipes of user /{user_id}/")
	reviews = get_user_reviews(user_id)
	all_recipes = []
	for index, row in reviews.iterrows():
		recipe_details = get_recipe_details(row["item"])
		print(recipe_details)
		all_recipes.append(recipe_details)
	l.info(f"All the recipes user /{user_id}/ /{all_recipes}/")
	return all_recipes


if __name__ == "__main__":
	print(get_all_recipes_of_user(0))
