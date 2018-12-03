import pandas as pd #sudo pip3 install pandas
import json #sudo pip3 install simplejson
import pickle #sudo pip3 install pickle 
import math

"""
This module reads the actual dataset and removes features that may not be required and dumps the dataframe into a file
"""

def load_dataset(path):
   
	JSON_DATA = ['device', 'geoNetwork', 'totals', 'trafficSource']

	data = pd.read_csv(path, dtype={'fullVisitorId': 'str', 'isMobile': int})

	for column in JSON_DATA:
		data = data.join(pd.DataFrame(data.pop(column).apply(pd.io.json.loads).values.tolist(), index=data.index))

	# Features that might not be required with greater certainity 
	DROP = ['browser', 'browserSize', 'browserVersion', 'deviceCategory', 'flashVersion', 'language', 'mobileDeviceBranding', 'mobileDeviceInfo',
		'mobileDeviceMarketingName', 'mobileDeviceModel', 'mobileInputSelector', 'operatingSystem', 'operatingSystemVersion', 'screenColors', 'screenResolution',
		'cityId', 'latitude', 'longitude', 'metro', 'networkDomain', 'networkLocation', 'adContent', 'adwordsClickInfo', 'campaign', 'campaignCode', 'isTrueDirect', 'keyword', 'medium', 'referralPath', 'source']

	# Features that might be required for time series based analysis
	REDUCE = ['date', 'city', 'country', 'region', 'subContinent']

	# Features that'll be required
	USE = []
	for col in data.columns:
		if((col not in DROP) and (col not in REDUCE)):
			USE.append(col)

	data = data[USE]

	# One hot encoding of categorical values
	CATEGORICAL_FEATURES = ['channelGrouping', 'socialEngagementType', 'continent']
	data = pd.concat([data,pd.get_dummies(data[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)],axis=1)
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
		data.set_value(i,'visitStartTime',data.get_value(i,'visitStartTime')%86400)

	return data

if(__name__ == "__main__"):

	# The initial dataset is present inside Raw_Data folder
	# The reduced dataset is present inside Processed_Data folder

	data = load_dataset("./Raw_Data/train.csv")
	outfile = open("./Processed_Data/train_processed.df-dump", "wb")
	pickle.dump(data, outfile)
	outfile.close()

	data = None
	data = load_dataset("./Raw_Data/test.csv")
	outfile = open("./Processed_Data/test_processed.df-dump", "wb")
	pickle.dump(data, outfile)
	outfile.close()


