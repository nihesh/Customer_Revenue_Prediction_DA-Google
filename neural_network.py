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

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

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

    file = open("./Processed_Data/train_processed.df-dump","rb")
    data = pickle.load(file)
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
    model = Sequential()
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(1, activation = "relu"))
    model.compile(optimizer = "rmsprop", loss="mean_squared_error", metrics=["accuracy"])

    # Feature selection - Find 2 features. 
    # reducer = PCA(n_components = 2)
    # Xtrain = reducer.fit_transform(Xtrain)
    # print("Explained variance of the two features")
    # print(reducer.explained_variance_ratio_)
    # Xtest = reducer.transform(Xtest)

    # training
    train_error=[]
    test_error=[]
    EPOCHS = 25

    for _ in range(EPOCHS):
		# result = model.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=1, batch_size=128, verbose=1) # 1.65 test Benchmark	
    	result = model.fit(Xtrain, Ytrain, validation_data=(Xtest,Ytest), epochs=1, batch_size=4096, verbose=1)
    	train_error.append(result.history["loss"][0])
    	test_error.append(result.history["val_loss"][0])

    plt.xlabel("Epochs")
    plt.ylabel("Training error")
    plt.plot(train_error,color="red",label="Training loss")
    plt.plot(test_error,color="green",label="Testing loss")
    plt.legend()
    plt.show()

    file = open("./Processed_Data/train_processed.df-dump","rb")
    data = pickle.load(file)
    Y = np.asarray(data['transactionRevenue'])
    dftrain, dftest, Ytrain, Ytest = train_test_split(data, Y, test_size=0.3, random_state = 45)
    

    print("Training complete")

    rev_pred = model.predict(Xtest)

    print("Training RMSE")
    getRMSE(dftrain, Ytrain, model.predict(Xtrain))
    
    rev_pred = model.predict(Xtest)
    
    print("Testing RMSE")
    getRMSE(dftest, Ytest, rev_pred)
