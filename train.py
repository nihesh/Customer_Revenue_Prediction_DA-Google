import pandas as pd #sudo pip3 install pandas
import pickle #sudo pip3 install pickle 
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import math
import numpy as np
import random


if(__name__ == "__main__"):

	file = open("./Processed_Data/train_processed.df-dump","rb")
	data = pickle.load(file)

	# Randomly subsampling from the dataset for faster computation
	# SIZE = 100000
	# data = data.ix[random.sample(list(data.index),SIZE)]

	DROP_ROWS = []
	for i in range(len(data.values)):
		if(data.get_value(i,'transactionRevenue') == 0):	# Removing points that don't lead to a transaction
			DROP_ROWS.append(i)

	data = data.drop(DROP_ROWS)

	# Target Label
	Y = np.asarray(data['transactionRevenue'])

	DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId']
	data = data.drop(DROP, axis = 1)
	data['visitStartTime'] = (data['visitStartTime']-data['visitStartTime'].mean())/data['visitStartTime'].std()

	# Input features
	X = data.values
	for i in range(len(X)):
		for j in range(len(X[0])):
			if(math.isnan(float(X[i][j]))):
				X[i][j] = 0
			else:
				X[i][j] = int(X[i][j])

	# Model Selection
	# model = LinearRegression(n_jobs = 4)
	# model = RandomForestRegressor(n_jobs = 4)
	# model = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(7,5), random_state=1)
	# model = SVR(kernel='linear')

	# Cross Validation
	cv_result = cross_validate(model, X, Y, cv = 3, scoring=('neg_mean_squared_error'), return_train_score = True)

	# Performance evaluation 
	print("Mean squared error of log revenue is ",end=" ")
	print(cv_result['test_score'])