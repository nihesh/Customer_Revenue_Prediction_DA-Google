import pickle 
import math
import numpy as np
from sklearn.cluster import KMeans

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
                
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X) # training clusters on training data
    y_pred = kmeans.labels_
    
    cluster = kmeans.predict(X_t) # clusters for test data
