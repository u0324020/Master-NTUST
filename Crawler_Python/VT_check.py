# coding by Jane
# -*- coding: UTF-8 -*-
from selenium.webdriver.chrome.options import Options
from multiprocessing import Queue
from selenium import webdriver
from bs4 import BeautifulSoup
import Levenshtein
import threading
import numpy as np
import requests
import urllib2
import time
import csv
import re
import httplib

VT_arr = []

def Index_Urlquery(Url):
	Id_counter = 0
	counter_num = 0
	page = urllib2.urlopen(Url).read()
	soup1 = BeautifulSoup(page,"html.parser")
	for table in soup1.find_all('table', {'class': 'test hideme'}):
		tds = table.find_all('a')
		for i in tds:
			counter_num = counter_num + 1
			Id_counter = Id_counter + 1
			ID_Column.append(Id_counter)
			url_get = i.get('title')
			if "https" in url_get:
				Url_Column.append(url_get)
				tar_url = url_get
			else:
				Url_Column.append('http://'+url_get)
				tar_url = 'http://'+url_get
			content_Urlquery(tar_url)
			if counter_num == 1 :
				break
			break
			href_dara = i.attrs['href']
			Index_Data.append(href_dara) # 最外層每個report
	return Index_Data


def VT_check(IP,Url):
	#https://www.virustotal.com/zh-tw/ip-address/138.201.146.246/information/
	Base_URL = "https://www.virustotal.com/zh-tw/ip-address/"
	End_URL = "/information/"
	Full_URL = Base_URL+IP+End_URL
	page = urllib2.urlopen(Full_URL).read()
	soup = BeautifulSoup(page,"html.parser")
	for VT_div in soup.find_all('div', {'id':'detected-urls'},{'style':'word-wrap:break-word;'}):
		for VT_A in VT_div.find_all('a', {'class':'margin-left-1'},{'target':'_blank'}):
			#distance = Levenshtein.ratio(Url,VT_A.renderContents())
			#print(distance)
			VT_arr.append( VT_A.renderContents().strip("'\n ") )
	if Url in VT_arr:
		return 1
	else :
		return 0

if __name__ == '__main__':
	Url_2 = "http://gxhwgcc.com/html/201805214490.html"
	Url = "124.227.108.202"
	
	print VT_check(Url,Url_2)










	