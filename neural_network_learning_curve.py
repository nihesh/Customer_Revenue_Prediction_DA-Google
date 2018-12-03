# Author: Nihesh Anderson
# File: neural_network.py

# Fixed seed to reproduce the results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import math
import numpy as np
from keras.layers import Dense,Conv2D, Flatten, Dropout
from keras.models import Sequential
import csv
from sklearn.metrics import mean_squared_error as MSE
from sklearn.decomposition import PCA
from copy import deepcopy
from matplotlib import pyplot as plt
from keras import backend as K
from keras.metrics import mse
from keras import regularizers
from sklearn.model_selection import learning_curve
from sklearn import metrics

if(__name__ == "__main__"):

    file = open("./Processed_Data/train_processed.df-dump","rb")
    data = pickle.load(file)
    data = data[:10000]
     # Target Label
    Y = np.asarray(data['transactionRevenue'])
    dftrain, dftest, Ytrain, Ytest = train_test_split(data, Y, test_size=0.3, random_state = 45)
    
    # training data dictionary with unique user ids
    fullVisitorId = np.asarray(dftest['fullVisitorId'])
    user_dict = {}
    for i in range(0, len(fullVisitorId)):
        if fullVisitorId[i] in user_dict.keys():
            user_dict[fullVisitorId[i]].append(i)
        else:
            user_dict[fullVisitorId[i]] = []
            user_dict[fullVisitorId[i]].append(i)
    
     # Target Label
    Y = np.asarray(data['transactionRevenue'])

    DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
    dftrain = dftrain.drop(DROP, axis = 1)
    dftest = dftest.drop(DROP, axis = 1)
    
     # Input features
    Xtrain = dftrain.values
    for i in range(len(Xtrain)):
        for j in range(len(Xtrain[0])):
            if(math.isnan(float(Xtrain[i][j]))):
                Xtrain[i][j] = 0
            else:
                Xtrain[i][j] = int(Xtrain[i][j])
                
    # Input features
    Xtest = dftest.values
    for i in range(len(Xtest)):
        for j in range(len(Xtest[0])):
            if(math.isnan(float(Xtest[i][j]))):
                Xtest[i][j] = 0
            else:
                Xtest[i][j] = int(Xtest[i][j])

    # Model Selection
    model = MLPRegressor((10,10,10), solver="lbfgs", early_stopping=True)

    # Plotting learning curve

    plt.figure()
    plt.title("Overfit / Underfit (MLP Regressor)")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, Xtrain, Ytrain, cv = 5, n_jobs = 4, train_sizes = np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes*100, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes*100, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

    # Learning curve ends

  
