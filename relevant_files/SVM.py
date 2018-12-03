# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:39:17 2018

@author: Pragya
"""

import pickle 
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import math
import numpy as np
import csv

if(__name__ == "__main__"):

    file = open("./Processed_Data/train_processed.df-dump","rb")
    data = pickle.load(file)
    
    file = open("./Processed_Data/test_processed.df-dump","rb")
    data_test = pickle.load(file)
    
    # training data dictionary with unique user ids
    fullVisitorId = np.asarray(data_test['fullVisitorId'])
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
    data = data.drop(DROP, axis = 1)
    DROP_TEST = ['fullVisitorId', 'sessionId', 'visitId', 'visitStartTime']
    data_test = data_test.drop(DROP_TEST, axis = 1)
    
	 # Input features
    X = data.values
    for i in range(len(X)):
        for j in range(len(X[0])):
            if(math.isnan(float(X[i][j]))):
                X[i][j] = 0
            else:
                X[i][j] = int(X[i][j])
                
    X_t = data_test.values
    for i in range(len(X_t)):
        for j in range(len(X_t[0])):
            if(math.isnan(float(X_t[i][j]))):
                X_t[i][j] = 0
            else:
                X_t[i][j] = int(X_t[i][j])
    
    
    #Classification first
    print (Y.shape)
    Y_clf = []
    for y in Y:
        if (y==0):
            Y_clf.append(-1)
        else:
            Y_clf.append(1)
                
    clf = SVC()
    clf.fit(X, Y_clf)
    
    #Find the training accuracy of the classifier
    acc = clf.score(X, Y_clf)
    print("Training Accuracy of SVM Classifier is ",acc)
    
    
	 # Model Selection
    model = Lasso(n_jobs = 4)

    # train error
    model.fit(X, Y)
    
    rev_pred = []
    for testSample in X_t:
        predClass = clf.predict(testSample)
        if(predClass==-1):
            rev_pred.append(predClass)
        else:
            rev_pred.append(model.predict(testSample))
    #rev_pred = model.predict(X_t)
    
    keys = list(user_dict.keys())
    sums = []
    
    submit = csv.reader(open("sample_submission.csv"))
    file = list(submit)
    
    for i in range (1, len(file)):
        list_ind = user_dict[file[i][0]]
        sum = 0
        if (len(list_ind) > 1):
            for j in range (0, len(list_ind)):
                sum = sum + math.exp(rev_pred[j])
            sums.append(sum)
            if (sum+1)<=0:
                sum = 0
            file[i][1] = float(math.log(sum + 1))
        else:
            sums.append(rev_pred[list_ind[0]])
            file[i][1] = rev_pred[list_ind[0]]
            
    with open('final.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(file)
            
    writeFile.close()