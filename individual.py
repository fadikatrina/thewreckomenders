from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from logger import l


# todo add explanations
# todo add evaluation
def knn(df_users, df_recipes, recipe_data):
	tags = recipe_data['tags']
	steps = recipe_data['steps']
	ingredients = recipe_data['ingredients']

	vectorizer = TfidfVectorizer(max_features=10000)
	tags = vectorizer.fit_transform(tags)
	steps = vectorizer.fit_transform(steps)
	ingredients = vectorizer.fit_transform(ingredients)

	df_recipes = df_recipes.rename(columns={'id': 'recipe_id'})
	df_users = df_users.rename(columns={'item': 'recipe_id'})
	df = pd.merge(df_users, df_recipes, on='recipe_id')
	df_sample = df.sample(n=1000)

	X = df_sample.iloc[:, [3, 4]]
	y = df_sample['rating']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	knn = KNeighborsRegressor(n_neighbors=3)
	knn.fit(X_train, y_train)

	y_pred_train = knn.predict(X_train)
	y_pred_test = knn.predict(X_test)

	# TODO compare expected outputs with actual outputs

	# fixme there is an error thrown here
	l.info('Accuracy on training data =', metrics.accuracy_score(np.array(y_train.to_list()), y_pred_train))
	l.info('Accuracy on testing data =', metrics.accuracy_score(np.array(y_test.to_list()), y_pred_test))
	l.info(metrics.classification_report(np.array(y_test.to_list()), y_pred_test))
