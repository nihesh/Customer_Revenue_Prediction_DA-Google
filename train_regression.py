import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import math
import numpy as np
from keras.layers import Dense,Conv2D, Flatten
from keras.models import Sequential
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

	 # Model Selection
    # model = MLPRegressor(hidden_layer_sizes = (10,10,10)) 1.7
    model = Sequential()
    model.add(Dense(30, activation = "relu"))
    model.add(Dense(30, activation = "relu"))
    model.add(Dense(30, activation = "relu"))
    model.add(Dense(30, activation = "relu"))
    model.add(Dense(1, activation = "relu"))
    model.compile(optimizer = "rmsprop", loss="mean_squared_error", metrics=["accuracy"])

    # train error
    model.fit(X, Y)
    rev_pred = model.predict(X_t)

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
            try:
                file[i][1] = rev_pred[list_ind[0]][0] 
            except:
                pass
            
    with open('final.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(file)
            
    writeFile.close()