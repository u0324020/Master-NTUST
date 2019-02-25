# coding by Jane
# -*- coding: UTF-8 -*-
from selenium.webdriver.chrome.options import Options
from multiprocessing import Queue
from selenium import webdriver
from bs4 import BeautifulSoup
import threading
import numpy as np
import requests
import urllib2
import time
import csv
import re
import httplib
#from test_CrawlertoCSV import get_url_dynamic2

# httplib.IncompleteRead exception 
httplib.HTTPConnection._http_vsn = 10
httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'
# normal
Target_Url = "https://urlquery.net/search?q=obfuscation"
Base_Url = "https://urlquery.net/"
Index_Data = []
CSV_data = []
VT_arr = []
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
Detection_Column = []
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
			url_get = i.get('title')
			if "https" in url_get:
				append_Url = url_get
			else:
				append_Url = 'http://'+url_get
			href_data = i.attrs['href']
			VT_check(href_data,append_Url)
			if jugement == 1:
				Id_counter = Id_counter + 1
				ID_Column.append(Id_counter)
				Url_Column.append(append_Url)
				Index_Data.append(href_data) # 最外層每個report
				print Id_counter
				Crawler(Index_Data)
				time.sleep(5)
			else: 
				print "None"


def VT_check(report,Url):
	global jugement
	jugement = 0
	IP_space = []
	inside_Urlquery(Base_Url+report)
	for trss in soup.find_all('tr', {'class':'even'}):
		for trs in trss.find_all('td'):
			A = trs.renderContents()
		IP_space.append(str(A.split(' ')))
		Tar_IP = str(IP_space[0])
	Full_IP = Tar_IP.strip("[']")
	Base_URL = "https://www.virustotal.com/zh-tw/ip-address/"
	End_URL = "/information/"
	Full_URL = Base_URL+str(Full_IP)+End_URL
	try:
		page = urllib2.urlopen(Full_URL).read()
		soup2 = BeautifulSoup(page,"html.parser")
		for VT_div in soup2.find_all('div', {'id':'detected-urls'},{'style':'word-wrap:break-word;'}):
			for VT_A in VT_div.find_all('a', {'class':'margin-left-1'},{'target':'_blank'}):
				VT_arr.append( VT_A.renderContents().strip("'\n ") )
	except:
		print "except"

	for i in VT_arr:
		if Url in i :
			jugement = 1

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

def Detection_Urlquery(soup):
	Det_data = []
	for Det_tr in soup.find_all('tr', {'class':'odd'}):
		for Det_trs in Det_tr.find_all('td', {'style':'text-align:left'}):
			Det_data = str(Det_trs.renderContents())
	if Det_data :
		Detection_Column.append(Det_data.strip("[']"))
	else :
		Detection_Column.append("None")

'''def BlackList_Urlquery(soup):
	Blk_data = []
	#for Blk_tr in soup.find_all('tr', {'class':'odd'}):
	for Blk_trs in soup.find_all('td', {'style':'text-align:center'}):
		Blk_data.append(Blk_trs.renderContents)
		#for Blk_trss in Blk_trs.find_all('td', {'style':'padding:0px;'}):
			
	print Blk_data

def Recent_Urlquery(soup):
	Rec_data = []
	Rec_counter = 0
	for Rec_tr in soup.find_all('tr', {'class':'even_highlight'}):
		for Rec_trs in Rec_tr.find_all('nobr'):
			Rec_counter = Rec_counter + 1
			print Rec_trs.renderContents()
			if Rec_counter == 10 :
				break'''

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

def Host_Urlquery(soup):
	#td class='wrapword'. pre style='font-size:14px;' 
	host_data = []
	for host_tr in soup.find_all('td', {'class':'wrapword'}):
		for host_trs in host_tr.find_all('pre', {'style':'font-size:14px;'},{'class':'wrapword'}):
			print host_trs
			
'''def content_Urlquery(url):
	print url
	global url_one
	global url_two
	global html_text
	opts = Options()
	opts.add_argument("Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.2.13) Gecko/20101203 Firefox/3.6.13")
	driver = webdriver.Chrome(chrome_options=opts)
	driver.get(url) 
	driver.find_element_by_xpath("//body").click()
	time.sleep(2)
	url_one = driver.current_url
	driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
	time.sleep(1)
	driver.execute_script('window.scrollTo(0, 0)')
	time.sleep(2)
	url_two = driver.current_url
	html_text=driver.page_source
	driver.quit()
	print html_text
	return html_text'''

def Crawler(Index_Data):
	print Index_Data
	print Base_Url+str(Index_Data[0])
	inside_Urlquery(Base_Url+str(Index_Data[0]))
	IP_Urlquery(soup)
	ASN_Urlquery(soup)
	Location_Urlquery(soup)
	Alert_Urlquery(soup)
	Detection_Urlquery(soup)
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
	writer.writerow(["ID","Url","IP","ASN","Location","Alert","Detection","Eval","Writes","Rediration"])
	writer.writerows(zip(ID_Column,Url_Column,IP_Column,ASN_Column,Location_Column,Alert_Column,Detection_Column,Eval_Column,Writes_Column,Redir_Column))
	file.close()
	print('Saving {} '.format(file_name))

if __name__ == '__main__':
	
	Index_Urlquery(Target_Url)
	#for path in Index_Data:
		#inside_Urlquery(Base_Url+path)
		#IP_Urlquery(soup)
		#VT_check(Base_Url+path)
		#print path
	File = "test"
	Write_CSV(File)
	'''
	url = "https://urlquery.net/report/54e6bf29-c27a-40d5-a9b0-4d3f8e37f6cb"
	inside_Urlquery(url)
	Host_Urlquery(soup)'''

	
