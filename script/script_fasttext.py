import fasttext
import pandas as pd
import spacy
from utils.logger import l
import string
import random
from tqdm import tqdm

nlp = None


def generate_fasttext_trained_model():
	model = fasttext.train_supervised('Data/processed/fasttext_training_enriched.txt')
	model.save_model("Data/processed/fasttext.bin")


def log_results(N, p, r):
	l.info("N\t" + str(N))
	l.info("P@{}\t{:.3f}".format(1, p))
	l.info("R@{}\t{:.3f}".format(1, r))


# todo:high move to eval
def test_model():
	l.info("Testing model")
	model = fasttext.load_model("Data/processed/fasttext.bin")
	log_results(*model.test("Data/processed/fasttext_testing_enriched.txt"))


def test_model_manually():
	global nlp
	l.info("Testing model manually")
	nlp = spacy.load('en_core_web_sm')
	model = fasttext.load_model("Data/processed/fasttext_quantized.ftz")
	while True:
		user_input = input("Enter your query ")
		l.info(f"User input: {user_input}")
		clean_user_input = spacy_process(user_input)
		result = model.predict(clean_user_input, k=5)
		l.info(f"prediction {result}")


def quantize_and_test():
	l.info("Qunatizing model")
	model = fasttext.load_model("Data/processed/fasttext.bin")
	model.quantize(input='Data/processed/fasttext_training_enriched.txt')
	l.info("Testing quantized model")
	log_results(*model.test("Data/processed/fasttext_testing_enriched.txt"))
	model.save_model("Data/processed/fasttext_quantized.ftz")


# todo:high extract this to general cleaning function in a helper
def spacy_process(text):
	try:
		doc = nlp(text)
	except ValueError:
		l.warning(f"Space does not understand '{text}' retunring empty string")
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

	# 	# Remove punctuation
	# for word in filtered_sentence:
	# 	if word in string.punctuation or word in ['"', ',', "[", "]"]:
	# 		filtered_sentence.remove(word)

	filtered_sentence = " ".join(filtered_sentence)
	return filtered_sentence.translate(str.maketrans('', '', string.punctuation))


def make_recipe_id_user_id_file():
	df_interaction = pd.read_csv('../Data/RAW_interactions.csv')
	df_interaction = df_interaction.loc[(df_interaction['rating'].astype(int) >= 4)]
	df_interaction['user_id'] = df_interaction['user_id'].apply(str)
	df_interaction = df_interaction.groupby(['recipe_id']).agg({'user_id': lambda x: ' '.join(x)})

	df_interaction.to_csv("Data/processed/fasttext_recipe_id_user_id.csv")


# todo:low get rid of global nlp and check instantiation in the method itself
def clean_and_export_recipes():
	global nlp
	"""
    For supervised training of fasttext, format of the file:
    __label__<recipe_id> __label__<contributor_id> <name> <tags> <steps> <description> <ingredients>
    """

	l.info("Preparing clean training file for fasttext")

	df_recipes = pd.read_csv('../Data/RAW_recipes.csv')
	df_recipes.drop(['minutes', 'submitted', 'nutrition', 'n_steps', 'n_ingredients'], axis=1, inplace=True)

	nlp = spacy.load('en_core_web_sm')

	no_recipes = len(df_recipes.index)
	clean_recipes = []
	for index, row in df_recipes.iterrows():
		l.debug(f"Currently doing {index} out of {no_recipes}")
		clean_recipes.append([
			f'__label__{row["id"]}',
			f'__label__{row["contributor_id"]}',
			str(spacy_process(row["name"])),
			str(spacy_process(row["tags"])),
			str(spacy_process(row["steps"])),
			str(spacy_process(row["description"])),
			str(spacy_process(row["ingredients"]))
		])

	l.info("Done with cleaning, exporting")

	random.shuffle(clean_recipes)
	train = clean_recipes[:int(no_recipes * 0.90)]
	test = clean_recipes[int(no_recipes * 0.90):]
	with open('Data/processed/fasttext_training.txt', 'w') as f:
		for item in train:
			f.write("%s\n" % " ".join(item))
	with open('Data/processed/fasttext_testing.txt', 'w') as f:
		for item in test:
			f.write("%s\n" % " ".join(item))


# todo:high differentiate between user and recipe labels for next trainings
def enrich_training_file_more_user_labels():
	df_interaction = pd.read_csv('Data/processed/fasttext_recipe_id_user_id.csv')

	with open('Data/processed/fasttext_training_enriched2.txt', 'w') as f:
		with open('Data/processed/fasttext_training.txt') as file:
			for line in tqdm(file):
				recipe_id = line.split(" ")[0]
				recipe_id = recipe_id[9:]
				if not recipe_id:
					continue
				new_labels = df_interaction.loc[df_interaction["recipe_id"] == int(recipe_id)][
					"user_id"].to_string().split(" ")
				if len(new_labels) > 5:
					new_labels = new_labels[4:]
					for new_label in new_labels:
						line = f"__label__{new_label} {line}"
					f.write("%s\n" % line)
				else:
					f.write("%s\n" % line)


if __name__ == "__main__":
	# clean_and_export_recipes()
	# make_recipe_id_user_id_file()
	# enrich_training_file_more_user_labels()
	# generate_fasttext_trained_model()
	# test_model()
	# quantize_and_test()
	test_model_manually()
