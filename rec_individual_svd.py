import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

'''
The SVD takes the recipe data, vectorizes the texts with TF-IDF and applies SVD to reduce the dimensionality
'''

def svd():
	recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')

	recipe_ids = recipe_data['id']
	minutes = recipe_data['minutes']
	tags = recipe_data['tags']
	nutrition = recipe_data['nutrition']
	steps = recipe_data['steps']
	ingredients = recipe_data['ingredients']

	vectorizer = TfidfVectorizer(max_features=500)
	X = {'tags':tags, 'steps':steps, 'ingredients':ingredients}
	X = pd.DataFrame(data=X)

	svd = TruncatedSVD(n_components=100, n_iter=10)
	svd.fit_transform(vectorizer.fit_transform(X['tags'] + X['steps'] + X['ingredients']))
	print(svd.explained_variance_ratio_)
	print(svd.explained_variance_ratio_.sum())

	return svd

if __name__ == "__main__":
	svd()
