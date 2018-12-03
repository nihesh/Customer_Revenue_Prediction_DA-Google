# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:51:05 2018

@author: Shravika Mittal
"""

import pickle 
import math
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

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
    
    return rmse

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
    
    print ("Data splitting done")
    
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
    
    print ("Data Processing done")
    
    print ("Showing the model is correct with varying max_leaf_nodes")
    leaves = [5, 15, 40, 70, 100, 200]
    train_arr = []
    test_arr = []
    labels = []
    X = []
    for i in leaves:
        
        labels.append("num_leaves = " + str(i))
        X.append(i)
        
        model = RandomForestRegressor(n_estimators = 100, max_depth = 8, max_leaf_nodes = i, max_features = 10)
        model.fit(Xtrain, Ytrain)
        print ("B")
    
        # Training RMSE
        print("Training RMSE")
        train_arr.append(getRMSE(dftrain, Ytrain, model.predict(Xtrain)))
        
        # Testing RMSE
        rev_pred = model.predict(Xtest)
        
        print("Testing RMSE")
        test_arr.append(getRMSE(dftest, Ytest, rev_pred))
        
    index = np.arange(len(labels))
    plt.bar(index - 0.3, train_arr, width = 0.3)
    plt.bar(index, test_arr, color = 'r', width = 0.3)
    plt.xticks(index, labels, fontsize = 8, rotation = 90)
    plt.show()
    
    plt.plot(X, train_arr, color = "red", label = "Training rmse")
    plt.plot(X, test_arr, color = "green", label = "Testing rmse")
    plt.legend()
    plt.xlabel("max_leaf_nodes")
    plt.ylabel("Rmse's")
    plt.show()