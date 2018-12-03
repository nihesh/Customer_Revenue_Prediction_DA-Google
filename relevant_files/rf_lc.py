# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:07:14 2018

@author: Shravika Mittal
"""

import pickle 
import math
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def getRMSE(dftest, rev_pred, Ytest):
    
    #dictionary of training data with unique user ids
    fullVisitorId = np.asarray(dftest['fullVisitorId'])
    user_dict = {}
    for i in range(0, len(fullVisitorId)):
        if fullVisitorId[i] in user_dict.keys():
            user_dict[fullVisitorId[i]].append(i)
        else:
            user_dict[fullVisitorId[i]] = []
            user_dict[fullVisitorId[i]].append(i)
            
    y_trans_rev = np.asarray(Ytest)
    ids =  list(user_dict.keys())
    y_test = {}
    
    #Actual values: Log of revenue sums
    for i in range (0, len(ids)):
        list_val = user_dict[ids[i]]
        sum = 0
        if (len(list_val) > 1):
            for j in range (0, len(list_val)):
                sum = sum + math.exp(y_trans_rev[list_val[j]])
            if sum <= 0:
                sum = 0
            y_test[ids[i]] = float(math.log(sum + 1))
        else:
            y_test[ids[i]] = y_trans_rev[list_val[0]]
    
    """ Dict of predicted values"""        
    y_trans_rev_pred = np.asarray(rev_pred)
    ids =  list(user_dict.keys())
    y_pred = {}
    
    #Predicted values: Log of revenue sums
    for i in range (0, len(ids)):
        list_val = user_dict[ids[i]]
        sum = 0
        if (len(list_val) > 1):
            for j in range (0, len(list_val)):
                sum = sum + math.exp(y_trans_rev_pred[list_val[j]])
            if sum <= 0:
                sum = 0
            y_pred[ids[i]] = float(math.log(sum + 1))
        else:
            y_pred[ids[i]] = y_trans_rev_pred[list_val[0]]
            
    """ Calculating RMSE """
    pred = []
    real = []
    for i in range (0, len(ids)):
        pred.append(y_pred[ids[i]])
        real.append(y_test[ids[i]])
        
    rmse = np.sqrt(MSE(real, pred))
        
    print("RMSE is ",rmse)

if(__name__ == "__main__"):

    file = open("./Processed_Data/train_processed_moreFeatures.df-dump","rb")
    data = pickle.load(file)
	 
    # Target Label
    Y = np.asarray(data['transactionRevenue'])
    dftrain, dftest, Ytrain, Ytest = train_test_split(data, Y, test_size=0.3, random_state = 45)
    
    #dictionary of training data with unique user ids
    fullVisitorId = np.asarray(dftest['fullVisitorId'])
    user_dict = {}
    for i in range(0, len(fullVisitorId)):
        if fullVisitorId[i] in user_dict.keys():
            user_dict[fullVisitorId[i]].append(i)
        else:
            user_dict[fullVisitorId[i]] = []
            user_dict[fullVisitorId[i]].append(i)
    
    DROP = ['transactionRevenue']
    dftrain = dftrain.drop(DROP, axis = 1)
    dftest = dftest.drop(DROP, axis = 1)
    
    print ("Data Splitting Done")
    
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
    
    print ("Data Processing Done")
    
    model = RandomForestRegressor(n_estimators = 100, max_depth = 8, max_leaf_nodes = 40, max_features = 10)
    model.fit(Xtrain, Ytrain)
    
    print ("Showing model does not underfit or overfit")
    
    # Training RMSE
    print("Training RMSE")
    getRMSE(dftrain, Ytrain, model.predict(Xtrain))
    
    # Testing RMSE
    rev_pred = model.predict(Xtest)
    
    print("Testing RMSE")
    getRMSE(dftest, Ytest, rev_pred)
    
    plt.figure()
    plt.title("Overfit / Underfit (Random Forest Classifier)")
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()