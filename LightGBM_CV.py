# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:59:59 2018

@author: Pragya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:45:40 2018

@author: Pragya
"""

import pickle
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import math
import numpy as np
import csv
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as MSE

def RFplotHyperparameterGraph(dftrain, dftest, Xtrain, Ytrain, Xtest, Ytest,d_train, paramsGS):
    #'max_depth': [1,2,3,4,5], 'min_samples_split': [2,3,4], 'min_samples_leaf': [2,4,6] 
    
    testingAcc = []
    trainingAcc = []
    FF = []
    MD = []
    BF = []
    labels = []
    X = []
    
    for i in paramsGS['feature_fraction']:
        for j in paramsGS['num_leaves']:
            for k in paramsGS['bagging_fraction']:
                FF.append(i)
                MD.append(j)
                BF.append(k)
                labels.append("num_leaves="+str(j))
                X.append(j)
                params = {
                    "objective" : "regression",
                    "metric" : "rmse",
                    #"boosting": "dart",
                    "lambda" : 0.8,
                    "num_leaves" : j,
                    "max_depth": 10,
                    "min_child_samples" : 100,
                    "learning_rate" : 0.01,
                    "bagging_fraction" : k,
                    "feature_fraction" : i,
                    "bagging_frequency" : 5,
                    "bagging_seed" : 2018,
                    "verbosity" : -1
                }
                model = lgb.train(params, d_train, 1000)
                
                Ypred = model.predict(Xtrain)
                print("Training RMSE")
                trainingAcc.append(getRMSE(dftrain, Ypred, Ytrain))
                Ypred = model.predict(Xtest)
                print("Testing RMSE")
                testingAcc.append(getRMSE(dftest, Ypred, Ytest))
    
    print (trainingAcc)
    print (testingAcc)
    index = np.arange(len(labels))
    plt.bar(index-0.3, trainingAcc, width = 0.3)
    plt.bar(index, testingAcc, color='r', width=0.3)
    plt.xticks(index, labels, fontsize=8, rotation=90)
    plt.show()
    
    

    plt.plot(X, trainingAcc,color="red",label="Training rmse")
    plt.plot(X, testingAcc,color="green",label="Testing rmse")
    plt.legend()
    plt.xlabel("Max_Depth")
    plt.ylabel("Rmse's")
    plt.show()

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
    
    
    
    #DROP = ['transactionRevenue', 'fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
    DROP = ['transactionRevenue']
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
    
    """
    paramsGS = {
            "max_depth": [15, 50, 100, 150, 200],
            "bagging_fraction": [0.8],
            "feature_fraction": [0.8]
    }
    
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"boosting": "dart",
        "lambda" : 0.8,
        "num_leaves" : 40,
        "min_child_samples" : 100,
        "learning_rate" : 0.01,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    model = lgb.LGBMRegressor(objective="regression", reg_lambda = 0.2, num_leaves=40, min_child_samples=100,learning_rate =0.01, bagging_frequency = 5, bagging_seed=2018, verbosity = -1)
    clf = GridSearchCV(model, param_grid = paramsGS, cv = 5, verbose = 1)
    
    print (clf.best_params_)
    print (clf.cv_results_)
    
    RFplotHyperparameterGraph(dftrain, dftest, Xtrain, Ytrain, Xtest, Ytest, d_train, paramsGS)
    
    """
    paramsGS = {
            "num_leaves": [5, 10, 30, 40, 80, 150, 200],
            "bagging_fraction": [0.8],
            "feature_fraction": [0.8]
            }
    RFplotHyperparameterGraph(dftrain, dftest, Xtrain, Ytrain, Xtest, Ytest, d_train, paramsGS)
        
    
    """
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"boosting": "dart",
        "lambda" : 0.8,
        "num_leaves" : 40,
        "max_depth":10,
        "min_child_samples" : 100,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.7,
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
    """