import pandas as pd #sudo pip3 install pandas
import pickle #sudo pip3 install pickle 
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math
import numpy as np
import random

if(__name__ == "__main__"):

	file = open("./Processed_Data/train_processed.df-dump","rb")
	data = pickle.load(file)

	# Randomly subsampling from the dataset for faster computation
	SIZE = 100000
	data = data.ix[random.sample(list(data.index),SIZE)]

	# Target Label
	Y = np.asarray(data['transactionRevenue'])
	for i in range(len(Y)):
		if(Y[i]!=0):
			Y[i] = 1	# If transaction happened, label as positive class 
	
	Y=Y.astype('int')

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

	# Feature selection - Find 2 features. 
	# reducer = PCA(n_components = 2)
	# reducer.fit(X)
	# print("Explained variance of the two features")
	# print(reducer.explained_variance_ratio_)

	# Model Selection
	model = LogisticRegression(n_jobs = 4)
	# model = RandomForestClassifier(n_jobs = 4, min_samples_leaf = 3)
	# model = MLPClassifier(alpha=1e-4, solver='adam', hidden_layer_sizes=(100,100), random_state=1, early_stopping = True)
	# model = SVC(C=0.5, kernel='linear')

	# Weight analysis	- Attempt 1
	# print()
	# print("Weight Analysis")
	# model.fit(X,Y)
	# for i in range(len(model.coef_[0])):
	# 	print(str(data.columns[i])+" "+str(model.coef_[0][i]))

	# Cross Validation
	cv_result = cross_validate(model, X, Y, cv = 5, scoring=('accuracy'), return_train_score = True)

	# Performance evaluation 
	print()
	print("Number of accurate classifications is ",end="")
	print((1-np.mean(cv_result['test_score']))*len(Y))
	print("Number of positive classes is ",end="")
	print(list(Y).count(1))
