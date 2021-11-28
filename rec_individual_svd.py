import csv

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

'''
The SVD takes the recipe data, vectorizes the texts with TF-IDF and applies SVD to reduce the dimensionality
'''


def getVectorizedData():
	np.random.seed(0)
	recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')

	recipe_ids = recipe_data['id']
	minutes = recipe_data['minutes']
	tags = recipe_data['tags']
	nutrition = recipe_data['nutrition']
	steps = recipe_data['steps']
	ingredients = recipe_data['ingredients']

	vectorizer = TfidfVectorizer(max_features=500)
	X = {'tags': tags, 'steps': steps, 'ingredients': ingredients}
	X = pd.DataFrame(data=X)

	return X, vectorizer


def findGoodNumberOfComponents():
	X, vectorizer = getVectorizedData()

	test_values = [1, 3, 5, 10, 50, 100, 200, 300]
	results = []

	with open('SVD_N_Components.csv', mode='w') as file:
		writer = csv.writer(file)

		for i in test_values:
			print(i)
			svd_result = svd(X, vectorizer, i)
			result = svd_result.explained_variance_ratio_.sum()
			print(result)

			writer.writerow(str(result))
			results.append(result)

	print(results)


def svd(X, vectorizer, components=100):
	# We're using a truncated SVD, since these are more efficient on sparse data and TF-IDF produces very sparse data
	svd = TruncatedSVD(n_components=components, n_iter=10, random_state=1)
	svd.fit_transform(vectorizer.fit_transform(X['tags'] + X['steps'] + X['ingredients']))
	# print(svd.explained_variance_ratio_)
	# print(svd.explained_variance_ratio_.sum())

	return svd


if __name__ == "__main__":
	X, vectorizer = getVectorizedData
	svd(X, vectorizer)
