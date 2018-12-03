# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:48:33 2018

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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

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
    
    
def TrainingCurve(Xtrain, Ytrain, Xtest, Ytest):
    train_error=[]
    test_error=[]
    EPOCHS = 25
    model = lgb.LGBMRegressor(objective="regression", reg_lambda = 0.8, num_leaves=40, max_depth=10, feature_fraction = 0.6, bagging_fraction=0.8, min_child_samples=100,learning_rate =0.01, bagging_frequency = 5, bagging_seed=2018, verbosity = -1)
    
    for _ in range(EPOCHS):
        
        result = model.fit(Xtrain, Ytrain, eval_set=(Xtest,Ytest), verbose=1)
        train_error.append(result.history["loss"][0])
        test_error.append(result.history["val_loss"][0])

    plt.xlabel("Boosting Stages")
    plt.ylabel("Training error")
    plt.plot(train_error,color="red",label="Training loss")
    plt.plot(test_error,color="green",label="Testing loss")
    plt.legend()
    plt.show()
    
def plot_CVscores(Xtrain, Ytrain, Xtest, Ytest):
    d_train = lgb.Dataset(Xtrain, Ytrain)
    d_valid = lgb.Dataset(Xtest, Ytest)
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"boosting_type": "goss",
        "lambda" : 0.8,
        "num_leaves" : 40,
        "max_depth":10,
        "min_child_samples" : 100,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    result = {}
    
    model = lgb.train(params, d_train, 1000, valid_sets = [d_valid, d_train], evals_result=result, verbose_eval=50)
    print(result.keys())
    trainingRMSE = (result['training']['rmse'])
    testingRMSE = (result["valid_0"]['rmse'])
    index = np.arange(1,1001,1)
    plt.plot(index,trainingRMSE, color="r",label="Training score")
    plt.plot(index,testingRMSE, color="g",label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt
    
    
def plot_Learning_Curve(Xtrain, Ytrain, ylim=None, cv=None, n_jobs=None):
    estimator = lgb.LGBMRegressor(objective="regression", reg_lambda = 0.8, num_leaves=40, max_depth=10, feature_fraction = 0.6, bagging_fraction=0.8, min_child_samples=100,learning_rate =0.01, bagging_frequency = 5, bagging_seed=2018, verbosity = -1)
    
    train_sizes=np.linspace(.1, 1.0, 5)
    
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, Xtrain, Ytrain, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print(train_sizes)
    train_sizes = train_sizes*100
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("means are ",train_scores_mean, test_scores_mean)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

if(__name__ == "__main__"):

    file = open("./Processed_Data/train_processed_moreFeatures.df-dump","rb")
    data = pickle.load(file)
    
	 # Target Label
    Y = np.asarray(data['transactionRevenue'])
    dftrain, dftest, Ytrain, Ytest = train_test_split(data[:100000][:], Y[:100000], test_size=0.3, random_state = 45)
    
    
    
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
                
    #plot_CVscores(Xtrain, Ytrain, Xtest, Ytest)
    
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_Learning_Curve(Xtrain, Ytrain, ylim=(0.0, 0.6), cv=cv, n_jobs=4)
    
    d_train = lgb.Dataset(Xtrain, Ytrain)
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"boosting_type": "goss",
        "lambda" : 0.8,
        "num_leaves" : 40,
        "max_depth":10,
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