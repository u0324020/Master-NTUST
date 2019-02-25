# coding by Jane
# -*- coding: UTF-8 -*-
from multiprocessing import Queue
from bs4 import BeautifulSoup
import threading
import requests
import urllib2
import time
import csv
import re
import httplib
# httplib.IncompleteRead exception 
httplib.HTTPConnection._http_vsn = 10
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'
# normal
Target_Url = "https://urlquery.net/search?q=obfuscation"
Base_Url = "https://urlquery.net/"
Index_Data = []
CSV_data = []
FileName = "Obfuscation"
# declare array_column
ID_Column = []
Url_Column = []
SecUrl_Column = []#
ThrUrl_Column = []#
IP_Column = []
ASN_Column = []
Location_Column = []
Alert_Column = []
Detection_Column = []#
BlackList_Column = []#
Recent_Column = []#
Eval_Column = []
Writes_Column = []
Redir_Column = []
Host_Column = []#
Conten_Column = []#


def Index_Urlquery(Url):
	Id_counter = 0
	page = urllib2.urlopen(Url).read()
	soup1 = BeautifulSoup(page,"html.parser")
	for table in soup1.find_all('table', {'class': 'test hideme'}):
		tds = table.find_all('a')
		for i in tds:
			Id_counter = Id_counter + 1
			ID_Column.append(Id_counter)
			url_get = i.get('title')
			if "https" in url_get:
				Url_Column.append(url_get.encode('utf-8'))
			else:
				Url_Column.append(('http://'+url_get).encode('utf-8'))
			href_dara = i.attrs['href']
			Index_Data.append(href_dara) # 最外層每個report
	return Index_Data

def inside_Urlquery(path_url): #內層的每個soup    
	global soup
	page = urllib2.urlopen(path_url).read()
	soup = BeautifulSoup(page,"html.parser")
	return soup

def IP_Urlquery(soup):
	IP_space = []
	for trss in soup.find_all('tr', {'class':'even'}):
		for trs in trss.find_all('td'):
			A = trs.renderContents()
		IP_space.append(str(A.split(' ')))
		Tar_IP = str(IP_space[0])
	IP_Column.append(Tar_IP.strip("[']"))# get IP address

def ASN_Urlquery(soup):
	for ASN_tr in soup.find_all('tr', {'class':'odd'})[1:2]:
		for ASN_trs in ASN_tr.find_all('td'):
			ASN_array = ASN_trs.renderContents()
			tar_ASN = ASN_array.strip("'")
	if ASN_array :		
		ASN_Column.append(ASN_array)# get ASN
	else:
		ASN_Column.append("None")
	

def Location_Urlquery(soup):
	for Lct_tr in soup.find_all('tr', {'class':'even'}):
		for Lct_trs in Lct_tr.find_all('img'):
			Lct_data = Lct_trs.get('title')
			Location_Column.append(str(Lct_data))


def Alert_Urlquery(soup):
	Alt_data = []
	for Alt_tr in soup.find_all('tr', {'class':'odd'}):
		for Alt_trs in Alt_tr.find_all('b', {'style':'color:red;'}):
			Alt_data.append(Alt_trs.renderContents())
	if Alt_data :		
		Alert_Column.append(str(Alt_data).strip("[ ]"))
	else:
		Alert_Column.append("None")

def Eval_Write_Urlquery(soup):
	p1 = re.compile(r'[(](.*?)[)]', re.S) 
	for Script_tr in soup.find_all('h3')[3:4]:
		Script_str = Script_tr.renderContents()
		Script_num = re.findall(p1, Script_str) 
		tar_script = str(map(int,Script_num)).strip('[ ]')
		if tar_script != '0' :
			for Eval_tr in soup.find_all('h3')[4:5]:
				Eval_str = Eval_tr.renderContents()
				Eval_num = re.findall(p1, Eval_str) 
				tar_Eval = str(map(int,Eval_num)).strip('[ ]')
				Eval_Column.append(" {}/{}".format(tar_Eval,tar_script))
			for Write_tr in soup.find_all('h3')[5:6]:
				Write_str = Write_tr.renderContents()
				Write_num = re.findall(p1, Write_str)
				tar_Write = str(map(int,Write_num)).strip('[ ]')
				Writes_Column.append(" {}/{}".format(tar_Write,tar_script))
		else:
			Eval_Column.append("None")
			Writes_Column.append("None")


def Rediration_Urlquery(soup):
	red_num = ""
	p1 = re.compile(r'[(](.*?)[)]', re.S) 
	for red_tr in soup.find_all('h2')[6:7]:
		red_str = red_tr.renderContents()
		red_num = re.findall(p1, red_str)
	if red_num :	
		tar_Redir = str((map(int,red_num)))	
		Redir_Column.append(tar_Redir.strip('[]'))
	else:
		Redir_Column.append("None")

def Crawler(soup):
	IP_Urlquery(soup)
	ASN_Urlquery(soup)
	Location_Urlquery(soup)
	Alert_Urlquery(soup)
	Eval_Write_Urlquery(soup)
	Rediration_Urlquery(soup)

def Write_CSV(FileName):
	now_date = time.strftime("%Y%m%d%H", time.localtime()) 
	file_name = FileName+("_")+now_date
	file = open(file_name + '.csv', 'wb')
	writer = csv.writer(file, ['URL'])
	'''	writer.writerow(["ID","Org_Url","Sec_Url","Thr_Url","IP","ASN",\
		"Location","Alert","Detection","BlackList_Counter","Recent_Time",\
		"Eval","Writes","Rediration","Host","Content"])
	writer.writerows(zip(ID_Column,Url_Column,SecUrl_Column,ThrUrl_Column\
		,IP_Column,ASN_Column,Location_Column,Alert_Column,Detection_Column,\
		BlackList_Column,Recent_Column,Eval_Column,Writes_Column,Redir_Column,\
		Host_Column,Conten_Column))'''
	writer.writerow(["ID","Url","IP","ASN","Location","Alert","Eval","Writes","Rediration"])
	writer.writerows(zip(ID_Column,Url_Column,IP_Column,ASN_Column,Location_Column,Alert_Column,Eval_Column,Writes_Column,Redir_Column))
	file.close()
	print('Saving {} '.format(file_name))

if __name__ == '__main__':
	Index_Urlquery(Target_Url)
	for path in Index_Data:
		inside_Urlquery(Base_Url+path)
		Crawler(soup)
		print path
	Write_CSV(FileName)
