import pickle 
from sklearn.linear_model import LinearRegression
import math
import numpy as np

if(__name__ == "__main__"):

    file = open("train_processed.df-dump","rb")
    data = pickle.load(file)
    
    file = open("test_processed.df-dump","rb")
    data_test = pickle.load(file)
    
    # training data dictionary with unique user ids
    fullVisitorId = np.asarray(data['fullVisitorId'])
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
    model = LinearRegression(n_jobs = 4)

    # train error
    model.fit(X, Y)
    rev_pred = model.predict(X)
    
    file = []
    keys = list(user_dict.keys())
    sums = []
    
    for i in range (len(user_dict.keys())):
        list_ind = user_dict[keys[i]]
        sum = 0
        for j in range (0, len(list_ind)):
            sum = sum + rev_pred[j]
        sums.append(sum)
        if (sum+1)<=0:
            sum = 0
        file.append(float(math.log(sum + 1)))
    
    with open('cum_sum.txt', 'w', newline='') as writeFile:
        for item in sums:
            writeFile.write("%s\n" % item)
            
    with open('train_pred.txt', 'w', newline='') as writeFile:
        for item in rev_pred:
            writeFile.write("%s\n" % item)
            
    with open('log_pred_train.txt', 'w', newline='') as writeFile:
        for item in file:
            writeFile.write("%s\n" % item)
            
        
    writeFile.close()