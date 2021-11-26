from helpers_data_load import df_users, nlp
from logger import l
import string


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


# Given string, returns it tokenized, lemmatized, filtered (stop words), without punctuation
def clean_text(text):
	try:
		doc = nlp(text)
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
		lexeme = nlp.vocab[word]
		if lexeme.is_stop == False:
			filtered_sentence.append(word)

	filtered_sentence = " ".join(filtered_sentence)
	return filtered_sentence.translate(str.maketrans('', '', string.punctuation))
