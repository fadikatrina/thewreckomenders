import fasttext
import pandas as pd
from logger import l
import random
from tqdm import tqdm
from helpers_general import clean_text
import settings
from eval_content_based import fasttext_auto_eval, show_words_labels


def generate_fasttext_trained_model():
	t_epoch = 20
	t_lr = 0.3
	l.info(f"Training of model started epoch /{t_epoch}/ leaning rate /{t_lr}/")
	model = fasttext.train_supervised('Data/processed/fasttext_training_enriched.txt', epoch=t_epoch, lr=t_lr)
	model.save_model(settings.FASTTEXT_MODEL_PATH_NORMAL)


def generate_fasttext_trained_model_auto_parameters():
	l.info(f"Training of model with autotune started")
	model = fasttext.train_supervised(input='Data/processed/fasttext_training_enriched.txt', autotuneValidationFile='Data/processed/fasttext_testing_enriched.txt')
	model.save_model(settings.FASTTEXT_MODEL_PATH_NORMAL_AUTO)


def quantize():
	l.info("Quantizing model")
	model = fasttext.load_model(settings.FASTTEXT_MODEL_PATH_NORMAL)
	model.quantize(input='Data/processed/fasttext_training_enriched.txt')
	model.save_model(settings.FASTTEXT_MODEL_PATH_QUANTIZED)


def make_recipe_id_user_id_file():
	l.info("Making the file with recipe_ids and user_ids that voted them 4 or higher")
	df_interaction = pd.read_csv('Data/RAW_interactions.csv')
	df_interaction = df_interaction.loc[(df_interaction['rating'].astype(int) >= 4)]
	df_interaction['user_id'] = df_interaction['user_id'].apply(str)
	df_interaction = df_interaction.groupby(['recipe_id']).agg({'user_id': lambda x: ' '.join(x)})

	df_interaction.to_csv("Data/processed/fasttext_recipe_id_user_id.csv")


# For supervised training of fasttext, format of the file:
# __label__user_<id> __label__recipe_<recipe_id> __label__contributor_<contributor_id> <name> <tags> <steps>
# <description> <ingredients>
def clean_and_export_recipes():
	l.info("Preparing clean recipes training and testing files")

	df_recipes = pd.read_csv('Data/RAW_recipes.csv')
	df_recipes.drop(['minutes', 'submitted', 'nutrition', 'n_steps', 'n_ingredients'], axis=1, inplace=True)

	with open("Data/processed/domain_stopwords.txt") as file:
		lines = file.readlines()
		domain_stopwords = [line.rstrip() for line in lines]

	with open('Data/processed/fasttext_training.txt', 'w') as f_train:
		with open('Data/processed/fasttext_testing.txt', 'w') as f_test:
			for index, row in tqdm(df_recipes.iterrows()):
				clean_recipe = [
					f'__label__recipe_{row["id"]}',
					f'__label__contributor_{row["contributor_id"]}',
					str(clean_text(row["name"], domain_stopwords)),
					str(clean_text(row["tags"], domain_stopwords)),
					str(clean_text(row["steps"], domain_stopwords)),
					str(clean_text(row["description"], domain_stopwords)),
					str(clean_text(row["ingredients"], domain_stopwords))
				]
				if random.random() < 0.1:
					f_train.write("%s\n" % " ".join(clean_recipe))
				else:
					f_test.write("%s\n" % " ".join(clean_recipe))

	l.info("Done with cleaning")


def enrich_training_file_more_user_labels():
	l.info("Enriching the testing and training files with more user_ids (users that liked the recipe)")
	df_interaction = pd.read_csv('Data/processed/fasttext_recipe_id_user_id.csv')

	with open('Data/processed/fasttext_training_enriched.txt', 'w') as f:
		with open('Data/processed/fasttext_training.txt') as file:
			enrich_file(f, file, df_interaction)

	with open('Data/processed/fasttext_testing_enriched.txt', 'w') as f:
		with open('Data/processed/fasttext_testing.txt') as file:
			enrich_file(f, file, df_interaction)


def enrich_file(f, file, df_interaction):
	for line in tqdm(file):
		recipe_id = line.split(" ")[0]
		recipe_id = recipe_id[9:]
		if not recipe_id:
			continue
		new_labels = df_interaction.loc[df_interaction["recipe_id"] == int(str(recipe_id).split("_")[1])][
			"user_id"].to_string().split(" ")
		if len(new_labels) > 5:
			new_labels = new_labels[4:]
			for new_label in new_labels:
				line = f"__label__user_{new_label} {line}"
			f.write("%s\n" % line)
		else:
			f.write("%s\n" % line)


if __name__ == "__main__":
	# clean_and_export_recipes()
	# make_recipe_id_user_id_file()
	# enrich_training_file_more_user_labels()
	# generate_fasttext_trained_model()
	# quantize()
	# fasttext_auto_eval()
	# show_words_labels()
	generate_fasttext_trained_model_auto_parameters()
