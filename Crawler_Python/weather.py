# coding: utf-8

import requests, lxml
import pandas as pd

df = pd.read_html("https://www.cwb.gov.tw/V7/forecast/town368/3Hr/6800300.htm")

head = ['Time', 'Temperature','Moisture','Rainfall%']

df = df[0].T #旋轉
df = df.iloc[1:] #去頭
df = df.drop([0,2,4,5,6,9],axis = 1) #刪欄位
df.columns = head

df.to_csv('weatherData.csv', index = False)