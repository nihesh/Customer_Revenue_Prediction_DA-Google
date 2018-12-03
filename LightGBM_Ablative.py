# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:44:48 2018

@author: Pragya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:45:40 2018

@author: Pragya
"""

import pickle 
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import math
import numpy as np
import csv
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as MSE

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
    
    
    
    #DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
    DROP = ['transactionRevenue', 'isMobile', 'hits', 'continent_cat', 'pageviews']
    dftrain = dftrain.drop(DROP, axis = 1)
    print (len(list(dftrain)))
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
    
    
    d_train = lgb.Dataset(Xtrain, Ytrain)
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"boosting_type": "goss",
        "lambda" : 0.8,
        "num_leaves" : 40,
        "max_depth":15,
        "min_child_samples" : 100,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    model = lgb.train(params, d_train, 1000)
    print("Training Complete")
    # model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    print("Training RMSE")
    getRMSE(dftrain, Ytrain, model.predict(Xtrain))
    
    rev_pred = model.predict(Xtest)
    
    print("Testing RMSE")
    getRMSE(dftest, Ytest, rev_pred)
