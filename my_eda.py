# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:24:30 2018

@author: Shravika Mittal
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def get_cluster(x_f, y_f, train):
    
    x_lab = x_f
    y_lab = y_f
    x_f = []
    y_f = []
    
    cont_val = np.zeros((len(train), 1))
    
    if x_lab == "continent":
        cont_af = train["continent_Africa"]
        cont_asia = train["continent_Asia"]
        cont_eu = train["continent_Europe"]
        cont_ocean = train["continent_Oceania"]
        cont_am = train["continent_Americas"]
        cont_n = train["continent_(not set)"]
        
        x_labels = ["A", "continent_Africa", "continent_Asia", "continent_Europe", "continent_Oceania", "continent_Americas", "continent_(not set)"]
        
        for i in range(0, len(cont_af)):
            if cont_af[i] == 1:
                cont_val[i] = 1
        for i in range(0, len(cont_asia)):
            if cont_asia[i] == 1:
                cont_val[i] = 2
        for i in range(0, len(cont_eu)):
            if cont_eu[i] == 1:
                cont_val[i] = 3
        for i in range(0, len(cont_ocean)):
            if cont_ocean[i] == 1:
                cont_val[i] = 4
        for i in range(0, len(cont_am)):
            if cont_am[i] == 1:
                cont_val[i] = 5
        for i in range(0, len(cont_n)):
            if cont_n[i] == 1:
                cont_val[i] = 6
            
        x_f = cont_val
        y_f = train[y_lab]
        
        x_f_new = []
        y_f_new = []
        for i in range (0, len(y_f)):
            if y_f[i] != 0:
                y_f_new.append(y_f[i])
                x_f_new.append(x_f[i])
                
        X = np.zeros((len(x_f_new), 2))
        X[:, 0] = x_f_new
        X[:, 1] = y_f_new
        
        kmeans = KMeans(n_clusters = 3)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        
        fig, ax = plt.subplots()
        
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        ax.set_xticklabels(x_labels, rotation = 90)
        plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, cmap = 'viridis')
        plt.show()
        
    else:    
        x_f = train[x_lab]
        y_f = train[y_lab]
        
        x_f_new = []
        y_f_new = []
        for i in range (0, len(y_f)):
            if y_f[i] != 0:
                y_f_new.append(y_f[i])
                x_f_new.append(x_f[i])
                
        X = np.zeros((len(x_f_new), 2))
        X[:, 0] = x_f_new
        X[:, 1] = y_f_new
        
        kmeans = KMeans(n_clusters = 3)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, cmap = 'viridis')
        plt.show()

def get_hist(x_f, train):
    
    rev = train[x_f]
    non_zero = []
    
    for i in range (0, len(rev)):
        if rev[i] != 0:
            non_zero.append(rev[i])
            
    plt.hist(non_zero, bins = 10)
    plt.xlabel("Distribution of transactionRevenue")
    plt.ylabel("Frequency")
    plt.show()

if(__name__ == "__main__"):
    
    file = open("./Processed_Data/train_processed.df-dump", "rb")
    train = pickle.load(file)
    
    print (list(train))
    
    # convert date for plotting graph
    dates = train["date"]
    conv_date = []
    for i in dates:
        new_date = pd.to_datetime(str(i)[0:4] + "-" + str(i)[4:6] + "-" + str(i)[6:8])
        conv_date.append(new_date)

    train["date"] = conv_date
    
    dates = train["date"]
    rev = train["transactionRevenue"]
    date_dict = {}
    for i in range (0, len(dates)):
        if dates[i] in date_dict.keys():
            date_dict[dates[i]] = date_dict[dates[i]] + 1
        else:
            date_dict[dates[i]] = 1
    x = []
    y = []
    keys = list(date_dict.keys())
    keys.sort()
    for i in keys:
        x.append(i)
        y.append(date_dict[i])
    plt.plot(x, y)
    plt.xlabel("Date of transaction")
    plt.ylabel("Number of transactions")
    plt.show()
    
    # cluster visualisation
    get_cluster("visitNumber", "transactionRevenue", train)
    get_cluster("hits", "transactionRevenue", train)
    get_cluster("continent", "transactionRevenue", train)
    
    # histogram visulalisation of transactionRevenue
    get_hist("transactionRevenue", train)
    
    # histogram visualization of transaction revenue for users
    ids = train["fullVisitorId"]
    rev = train["transactionRevenue"]
    trans_dict = {}
    for i in range (0, len(ids)):
        if ids[i] in trans_dict.keys():
            trans_dict[ids[i]] = trans_dict[ids[i]] + rev[i]
        else:
            trans_dict[ids[i]] = rev[i] 
            
    revs = list(trans_dict.values())
    non_zero = []
    pay = 0
    for i in range(0, len(revs)):
        if revs[i] != 0:
            non_zero.append(revs[i])
            pay = pay + 1
    plt.hist(non_zero, bins = 10)
    plt.xlabel("Distribution of transactionRevenue as per unique users")
    plt.ylabel("Frequency")
    plt.show()
    
    # user data
    print ("Unique Users: " + str(len(revs)))
    print ("Paying Users: " + str(pay))
        
    # number of zero and non-zero transactions
    rev = train["transactionRevenue"]
    count_zero = 0
    count_non = 0
    for i in range(0, len(rev)):
        if rev[i] == 0:
            count_zero = count_zero + 1
        else:
            count_non = count_non + 1
    print ("Non zero transactionRevenue: " + str(count_non))
    print ("Zero transactionRevenue: " + str(count_zero))
    
    Y = np.asarray(train['transactionRevenue'])
    
    x_train, x_test, y_train, y_test = train_test_split(train, Y, test_size = 0.3, random_state = 45)
    
    num_train = len(x_train)
    print ("Train Size: " + str(num_train))
    
    num_test = len(x_test)
    print ("Test Size: " + str(num_test))