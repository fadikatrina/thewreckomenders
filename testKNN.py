from rec_individual_knn import knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from logger import l

df_recipes = None
import pandas as pd

def test(df_users, recipe_data):
	global df_recipes
	# data loading usually in `helpers_data_load.py` but since this takes forever, would prefer to run it only if necessary
	# since its global will run only once per application cycle even if knn is run multiple times
	if not df_recipes:
		df_recipes = pd.read_csv('Data/processed/knn_recipes.csv')

	df_recipes = recipe_data.rename(columns={'id': 'recipe_id'})
	# df_recipes = df_recipes.rename(columns={'id': 'recipe_id'})
	df_users = df_users.rename(columns={'item': 'recipe_id'})
	df = pd.merge(df_users, df_recipes, on='recipe_id')
	sample_size = 100000
	df_sample = df.sample(n=sample_size)

	titles = list(df_sample.columns)
	feature1 = 5
	feature2 = 10
	X = df_sample.iloc[:, [feature1, feature2]]  # training data (nrs of steps and nr of ingredients)
	y = df_sample['rating']  # predicted value
	y = y.apply(np.int64)  # convert to int

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	l.info('sample size =')
	l.info(sample_size)
	l.info('features = ')
	l.info(titles[feature1])
	l.info(titles[feature2])

	for i in range(1,100):
		k = i
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)

		y_pred_train = knn.predict(X_train)
		y_pred_test = knn.predict(X_test)

		l.info('k =')
		l.info(k)

		l.info('Accuracy on training data =')
		l.info(metrics.accuracy_score(np.array(y_train.to_list()), y_pred_train))
		l.info('Accuracy on testing data =')
		l.info(metrics.accuracy_score(np.array(y_test.to_list()), y_pred_test))
		l.info(metrics.classification_report(np.array(y_test.to_list()), y_pred_test))

if __name__ == "__main__":
	recipe_data = pd.read_csv('Data/RAW_recipes.csv', sep=',')
	df_users = pd.read_csv('Data/processed/knn_users.csv')
	#knn(df_users, recipe_data)
	test(df_users, recipe_data)
