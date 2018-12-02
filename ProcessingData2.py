# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:32:50 2018

@author: Pragya
"""

import pandas as pd #sudo pip3 install pandas
import json #sudo pip3 install simplejson
import pickle #sudo pip3 install pickle 
import math

"""
This module reads the actual dataset and removes features that may not be required and dumps the dataframe into a file
"""

def load_dataset(path):
   
    JSON_DATA = ['device', 'geoNetwork', 'totals', 'trafficSource']
    data = pd.read_csv(path, dtype={'fullVisitorId': 'str', 'isMobile': int, 'visits':int, "hits": int, "pageviews": int, "bounces": int, "newVisits": int, 'date': int})
    
    for column in JSON_DATA:
        data = data.join(pd.DataFrame(data.pop(column).apply(pd.io.json.loads).values.tolist(), index=data.index))

    # Features that might not be required with greater certainity 
    DROP = ['browserSize', 'browserVersion', 'flashVersion', 'language', 'mobileDeviceBranding', 'mobileDeviceInfo',
		'mobileDeviceMarketingName', 'mobileDeviceModel', 'mobileInputSelector', 'operatingSystemVersion', 'screenColors', 'screenResolution', 'socialEngagementType',
		'cityId', 'latitude', 'longitude', 'metro', 'networkLocation', 'adContent', 'adwordsClickInfo', 'campaign', 'campaignCode', 'isTrueDirect', 'keyword', 'medium', 'referralPath']

    # Features that might be required for time series based analysis
    REDUCE = []

    # Features that'll be required
    USE = []
    for col in data.columns:
        if((col not in DROP) and (col not in REDUCE)):
            USE.append(col)

    data = data[USE]

    # One hot encoding of categorical values
    CATEGORICAL_FEATURES = ['browser', 'operatingSystem', 'deviceCategory', 'channelGrouping', 'networkDomain', 'continent', 'city', 'country', 'region', 'subContinent', 'source']
    for feature in CATEGORICAL_FEATURES:
        data[feature] = data[feature].astype('category')
        s = feature + "_cat"
        data[s] = data[feature].cat.codes
      
    data = data.drop(CATEGORICAL_FEATURES, axis=1)

	# Converting isMobile from bool to int and finding log revenue
    for i in range(len(data.values)):
        try:
            if(math.isnan(float(data.get_value(i,'transactionRevenue'))) or float(data.get_value(i,'transactionRevenue')) == 0):
                data.set_value(i,'transactionRevenue',0)
            else:
                data.set_value(i,'transactionRevenue',float(math.log(float(data.get_value(i,'transactionRevenue')))))
        except:
            pass
        data.set_value(i,'isMobile',int(data.get_value(i,'isMobile')))

    print (list(data))
    return data

if(__name__ == "__main__"):

	# The initial dataset is present inside Raw_Data folder
	# The reduced dataset is present inside Processed_Data folder

	data = load_dataset("./Raw_Data/train.csv")
	outfile = open("./Processed_Data/train_processed_moreFeatures.df-dump", "wb")
	pickle.dump(data, outfile)
	outfile.close()

