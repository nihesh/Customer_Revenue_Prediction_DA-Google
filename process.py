import pandas as pd #sudo pip3 install pandas
import json #sudo pip3 install simplejson
import pickle #sudo pip3 install pickle 

"""
This module reads the actual dataset and removes features that may not be required and dumps the dataframe into a file
"""

def load_dataset(path):
   
	JSON_DATA = ['device', 'geoNetwork', 'totals', 'trafficSource']

	data = pd.read_csv(path, dtype={'fullVisitorId': 'str'})

	for column in JSON_DATA:
		data = data.join(pd.DataFrame(data.pop(column).apply(pd.io.json.loads).values.tolist(), index=data.index))

	# Features that might not be required with greater certainity 
	DROP = ['browser', 'browserSize', 'browserVersion', 'deviceCategory', 'flashVersion', 'language', 'mobileDeviceBranding', 'mobileDeviceInfo',
		'mobileDeviceMarketingName', 'mobileDeviceModel', 'mobileInputSelector', 'operatingSystem', 'operatingSystemVersion', 'screenColors', 'screenResolution',
		'cityId', 'latitude', 'longitude', 'metro', 'networkDomain', 'networkLocation', 'adContent', 'adwordsClickInfo', 'campaign', 'campaignCode', 'isTrueDirect', 'keyword', 'medium', 'referralPath', 'source']

	# Features that might be required for time series based analysis
	REDUCE = ['visits', 'date']

	# Features that'll be required
	USE = []
	for col in data.columns:
		if((col not in DROP) and (col not in REDUCE)):
			USE.append(col)

	data = data[USE]
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


