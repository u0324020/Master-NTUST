# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:24:04 2019

@author: hyc
"""
import pandas as pd
import pickle
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib

class_dir  = ["EITest/", "Emotet/", "Hancitor/", "Nuclear/", "Rig/", "TrickBot/", "Dridex/", "Razy/", "HTBot/", "wannacry/", "normal/"]
labels = ["EITest", "Emotet", "Hancitor", "Nuclear", "Rig", "TrickBot", "Dridex", "Razy", "HTBot", "wannacry", "normal"]
feature_names = ["protocol", "dest port", "bytes_out", "num_pkts_out", "bytes_in", "num_pkts_in", "duration", "ip out ttl", "ip in ttl"]
analysis = "5_analysis/"

for i in range(256):
	name = "byte_" + str(i)
	feature_names.append(name)
	
def preprocess_data():

	train_X_normal, train_X_malicious = None, None

	for class_num, class_name in enumerate(class_dir):
		print(class_num)
		train_data = np.array(pickle.load(open("2_Pkl/train/" + class_name[:-1] + ".pkl", "rb")))
		train_label = np.array([class_num for i in range(len(train_data))])
		
		if class_name == "normal/":
			train_X_normal =  train_data
			continue
		
		if class_num == 0:
			train_X_malicious = train_data			
		else:					
			train_X_malicious = np.concatenate((train_X_malicious, train_data), axis=0)			
			
	
	train_X_normal = np.nan_to_num(train_X_normal)
	train_X_malicious = np.nan_to_num(train_X_malicious)	
	
	print(train_X_normal.shape)
	print(train_X_malicious.shape)
	
	train_X_normal = pd.DataFrame(train_X_normal, columns=feature_names)
	train_X_malicious = pd.DataFrame(train_X_malicious, columns=feature_names)

	return train_X_normal, train_X_malicious

train_X_normal, train_X_malicious = preprocess_data()

list_dir = ["dest port", "ip in ttl", "bytes_out", "num_pkts_out", "bytes_in", "num_pkts_in", "duration", "byte_0", "byte_1", "byte_3"]

train_X_normal["duration"] = train_X_normal["duration"].apply(lambda x : int(x))
train_X_malicious["duration"] = train_X_malicious["duration"].apply(lambda x : int(x))

train_X_normal["duration"] = train_X_normal["duration"].apply(lambda x : max(0, x))
train_X_malicious["duration"] = train_X_malicious["duration"].apply(lambda x : max(0, x))

for i, feature_name in enumerate(list_dir):
	normal_count = pd.value_counts(train_X_normal[feature_name])
	normal_count = normal_count.sort_index()
	
	malicious_count = pd.value_counts(train_X_malicious[feature_name])
	malicious_count = malicious_count.sort_index()
	
	fig = plt.figure(num=i, figsize=(16, 9)) 
		
	plt.xlabel(feature_name + " value")
	plt.ylabel("number")	
	
	bins = [i for i in range(500)]
	normal_y = []
	malicious_y = []
	
	for i in range(500):
		if i in list(normal_count.index):
			normal_y.append(normal_count.get(i))
		else:
			normal_y.append(0)
			
	for i in range(500):
		if i in list(malicious_count.index):
			malicious_y.append(malicious_count.get(i))
		else:
			malicious_y.append(0)
			
	#plt.hist([x, y], bins=bins, stacked=True)
	
	plt.bar(bins, normal_y, alpha=1, color="orange",  label="normal") 
	plt.bar(bins, malicious_y, alpha=0.5, color="blue", label="malicious") 
	plt.legend(loc="upper right")
	
	#plt.show() 
	filename = analysis + feature_name + '.png' 
	plt.savefig(filename)
	
	#break















