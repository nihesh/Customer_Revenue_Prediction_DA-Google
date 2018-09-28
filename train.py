import pandas as pd #sudo pip3 install pandas
import pickle #sudo pip3 install pickle 
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import math
import numpy as np

if(__name__ == "__main__"):

	file = open("./Processed_Data/train_processed.df-dump","rb")
	data = pickle.load(file)

	# Target Label
	Y = np.asarray(data['transactionRevenue'])

	DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
	data = data.drop(DROP, axis = 1)
	# Input features
	X = data.values
	for i in range(len(X)):
		for j in range(len(X[0])):
			if(math.isnan(float(X[i][j]))):
				X[i][j] = 0
			else:
				X[i][j] = int(X[i][j])

	# Model Selection
	model = LinearRegression(n_jobs = 4)

	# Cross Validation
	cv_result = cross_validate(model, X, Y, cv = 3, scoring=('neg_mean_squared_error'), return_train_score = True)

	# Performance evaluation 
	print("Mean squared error of log revenue is ",end=" ")
	print(cv_result['test_score'])