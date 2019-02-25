# -*- coding: UTF-8 -*-
#coding by Jane 
import requests
import numpy as np
import urllib2
import urllib
from multiprocessing import Queue
from bs4 import BeautifulSoup
import threading
import time
import csv
import re
import httplib

VT_Url = []
Org_Url = []
def Open_file(filename):
	file = open(filename+'.txt', 'r')
	lines=file.readlines()
	for i in lines:
		VT_search(i)

def VT_search(Url):
	print Url
	page = urllib.urlopen(Url).read()
	soup = BeautifulSoup(page,"html.parser")
	for divv in soup.find_all('div',{'id':'detected-urls'}):
		for div in divv.find_all('div', {'class':'enum'}):
			for divs in div.find_all('a'):
				VT_Url.append(divs.renderContents())

def save_file():
    file_name = "naughtyseductions.blogspot.in"
    file = open(file_name + '.csv', 'wb')
    writer = csv.writer(file, ['URL'])
    c = 0
    for i in VT_Url:
    	writer.writerow([i])
    	c = c + 1
    print c
    file.close()
    print('Saving {} '.format(file_name))
				


if __name__ == '__main__':
	filename = '1'
	Open_file(filename)
	save_file()
