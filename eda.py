import pandas as pd #sudo pip3 install pandas
import pickle #sudo pip3 install pickle 
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import math
import numpy as np
import random

if(__name__ == "__main__"):

	file = open("./Processed_Data/train_processed.df-dump","rb")
	data = pickle.load(file)

	# Random subsampling
	# SIZE = 10000
	# data = data.ix[random.sample(list(data.index),SIZE)]

	# Drop negative classes
	DROP_ROWS = []
	# for i in range(len(data.values)):
	# 	if(data.get_value(i,'transactionRevenue') == 0):	# Removing points that don't lead to a transaction
	# 		DROP_ROWS.append(i)
	# data = data.drop(DROP_ROWS)

	# Target Label
	Y = np.asarray(data['transactionRevenue'])
	for i in range(len(Y)):
		if(Y[i]!=0):
			Y[i] = 1	# If transaction happened, label as positive class 
	
	Y=Y.astype('int')

	DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
	POSIX_DATA = list(data['visitStartTime'])
	data = data.drop(DROP, axis = 1)
	
	# Input features
	X = data.values
	for i in range(len(X)):
		for j in range(len(X[0])):
			if(math.isnan(float(X[i][j]))):
				X[i][j] = 0
			else:
				X[i][j] = int(X[i][j])

	reduced_data = PCA(n_components = 2)
	X = reduced_data.fit_transform(X)
	print(reduced_data.explained_variance_ratio_)

	neg_class = [[],[]] 
	pos_class = [[],[]]

	for i in range(len(Y)):
		if(Y[i] == 0):
			neg_class[0].append(X[i][0])
			neg_class[1].append(X[i][1])
		else:
			pos_class[0].append(X[i][0])
			pos_class[1].append(X[i][1])

	# Plotting class vs features
	plt.scatter(neg_class[0], neg_class[1], cmap="viridis")
	plt.scatter(pos_class[0], pos_class[1], cmap="viridis")
	plt.xlabel("Feature 1 (61%)")
	plt.ylabel("Feature 2 (37%)")
	plt.show()

	# Analysing posix distribution for positive classes. Seems useful
	# POSIX_DATA = np.asarray(POSIX_DATA)%86400
	# plt.hist(POSIX_DATA)
	# plt.show()
