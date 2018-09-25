import pandas as pd #sudo pip3 install pandas
import pickle #sudo pip3 install pickle 
from matplotlib import pyplot as plt
import math
import numpy as np

"""
This module examines distribution of data and feature-output relationship
"""

if(__name__ == "__main__"):

	file = open("./Processed_Data/train_processed.df-dump","rb")
	data = pickle.load(file)
	revenue = list(data["transactionRevenue"])	
	other = []
	for i in range(len(revenue)):
		if(math.isnan(float(revenue[i]))):
			revenue[i] = 0
		else:
			other.append(int(revenue[i]))
			revenue[i] = int(revenue[i])

	print(np.mean(np.asarray(other)))
	print(np.std(np.asarray(other)))
	print(np.mean(np.asarray(revenue)))
	print(np.std(np.asarray(revenue)))