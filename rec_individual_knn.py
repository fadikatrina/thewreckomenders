from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from logger import l

df_recipes = None


# todo:high add explanations
# todo: use SVD
# todo: experimenting the kNN with
# - different k
# - different size of sample
# - different features

def knn(df_users, recipe_data):
	global df_recipes
	# data loading usually in `helpers_data_load.py` but since this takes forever, would prefer to run it only if necessary
	# since its global will run only once per application cycle even if knn is run multiple times
	if not df_recipes:
		df_recipes = pd.read_csv('Data/processed/knn_recipes.csv')

	tags = recipe_data['tags']
	steps = recipe_data['steps']
	ingredients = recipe_data['ingredients']

	vectorizer = TfidfVectorizer(max_features=10000)
	tags = vectorizer.fit_transform(tags)
	steps = vectorizer.fit_transform(steps)
	ingredients = vectorizer.fit_transform(ingredients)

	df_recipes = recipe_data.rename(columns={'id': 'recipe_id'})
	#df_recipes = df_recipes.rename(columns={'id': 'recipe_id'})
	df_users = df_users.rename(columns={'item': 'recipe_id'})
	df = pd.merge(df_users, df_recipes, on='recipe_id')
	sample_size = 10000
	df_sample = df.sample(n=sample_size)

	X = df_sample.iloc[:, [5, 14]] # training data (nrs of steps and nr of ingredients)
	y = df_sample['rating'] # predicted value
	y = y.apply(np.int64) # convert to int

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	k = 3
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)

	y_pred_train = knn.predict(X_train)
	y_pred_test = knn.predict(X_test)

	# todo:high compare expected outputs with actual outputs
	# todo:high add evaluation
	X_new = [[10, 90]]  # [nrs of steps, nr of ingredients]
	# [1,10] gives 4
	# [1, 15] gives 4
	y_predict = knn.predict(X_new)

	print('k=', k)
	print('sample size=', sample_size)
	print('features = nrs of steps and nr of ingredients')
	print('---------------------')
	print('For a recipes that has a calorie level of', X_new[0][0], 'and takes', X_new[0][1],
		  'minutes to cook, gets a rating of', y_predict[0])

	print('Accuracy on training data =', metrics.accuracy_score(np.array(y_train.to_list()), y_pred_train))
	print('Accuracy on testing data =', metrics.accuracy_score(np.array(y_test.to_list()), y_pred_test))
	print('')
	print(metrics.classification_report(np.array(y_test.to_list()), y_pred_test))

	l.info(metrics.accuracy_score(np.array(y_train.to_list()), y_pred_train))
	l.info(metrics.accuracy_score(np.array(y_test.to_list()), y_pred_test))
	l.info(metrics.classification_report(np.array(y_test.to_list()), y_pred_test))

