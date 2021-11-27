import pandas as pd
from utils.logger import l
from sklearn.model_selection import train_test_split

# model = gensim.models.Word2Vec.load("modelName.model")
# model = model.init_sims()
# model.save("modelName.model")

'''
# DOES NOTHING, STARTED WORKING ON IT THEN OPTED FOR FASTTEXT
'''


def generate_trained_model():
	l.info("Started generating of trained word2vec model")
	df_recipes = pd.read_csv('../Data/RAW_recipes.csv', sep=',')
	df_interactions = pd.read_csv('../Data/RAW_interactions.csv', sep=',')

	l.info(f"missing values from df_recipes {df_recipes.isnull().sum()}")
	l.info(f"missing values from df_interactions {df_interactions.isnull().sum()}")

	train, test = train_test_split(df_interactions, test_size=0.2)


if __name__ == "__main__":
	generate_trained_model()
