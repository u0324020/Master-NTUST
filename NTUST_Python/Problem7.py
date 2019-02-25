#problem seven coding by Jane
#assignment:BeautifulSoup,CSV,Thread
from multiprocessing import Queue
from bs4 import BeautifulSoup
import threading
import requests
import urllib2
import time
import csv
import time

base_url = "https://www.phishtank.com/"

def job1_scraped_data(Valid_value,Active_value):
	#index
	Data_num = 0
	URL_DATA = []
	URL_Vonline = 'https://www.phishtank.com/phish_search.php?valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'
	page = urllib2.urlopen(URL_Vonline).read()
	soup = BeautifulSoup(page,"html.parser")	
	for tr in soup.find_all('tr')[2:]:
	    tds = tr.find_all('td')
	    P_ID = tds[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
	    for i in P_ID:
			mytxt = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
			link_soup = BeautifulSoup(mytxt,'lxml')
			mylink = link_soup.find('a')
			link_url = mylink.attrs['href']
			full_url = base_url + link_url
			#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
			full_resp = requests.get(full_url)
			full_url_txt = full_resp.text
			full_soup = BeautifulSoup(full_url_txt, "lxml")
			for that_span in full_soup.findAll('span',{'style':'word-wrap:break-word;'}):
				target_url = that_span.find('b').text
				URL_DATA.append(target_url)
				#print target_url
				Data_num = Data_num+1
				
	#page2....
	for i in range(0, 9, 1):
		URL_Vonline_Page = 'https://www.phishtank.com/phish_search.php?page='+str(i)+'&valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'	
		page2 = urllib2.urlopen(URL_Vonline_Page).read()
		soup2 = BeautifulSoup(page2,"html.parser")	
		for tr2 in soup2.find_all('tr')[2:]:
		    tds2 = tr2.find_all('td')
		    P_ID2 = tds2[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
		    for i in P_ID2:
				mytxt2 = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
				link_soup2 = BeautifulSoup(mytxt2,'lxml')
				mylink2 = link_soup2.find('a')
				link_url2 = mylink2.attrs['href']
				full_url2 = base_url + link_url2
				#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
				full_resp2 = requests.get(full_url2)
				full_url_txt2 = full_resp2.text
				full_soup2 = BeautifulSoup(full_url_txt2, "lxml")
				for that_span2 in full_soup2.findAll('span',{'style':'word-wrap:break-word;'}):
					target_url2 = that_span2.find('b').text
					URL_DATA.append(target_url2)
					contrast_URL = str(URL_DATA[Data_num-1])
					if target_url2 == contrast_URL:
						check_can_save = False
						URL_DATA.pop(Data_num)
						print (("* T1 Double Data in No.%d *\n")%Data_num)
					else:
						Data_num = Data_num+1
#Return Data
	file_name = 'Valid+Online_download'
	file = open(file_name + '.csv', 'wb')
	writer = csv.writer(file, ['Phishing_URL'])
	writer.writerow(["Phish URL"])
	for val in URL_DATA:
		writer.writerow([val])
	file.close()
	print('Saving {} '.format(file_name))

def job2_scraped_data(Valid_value,Active_value):
	#index
	Data_num = 0
	URL_DATA = []
	URL_Vonline = 'https://www.phishtank.com/phish_search.php?valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'
	page = urllib2.urlopen(URL_Vonline).read()
	soup = BeautifulSoup(page,"html.parser")	
	for tr in soup.find_all('tr')[2:]:
	    tds = tr.find_all('td')
	    P_ID = tds[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
	    for i in P_ID:
			mytxt = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
			link_soup = BeautifulSoup(mytxt,'lxml')
			mylink = link_soup.find('a')
			link_url = mylink.attrs['href']
			full_url = base_url + link_url
			#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
			full_resp = requests.get(full_url)
			full_url_txt = full_resp.text
			full_soup = BeautifulSoup(full_url_txt, "lxml")
			for that_span in full_soup.findAll('span',{'style':'word-wrap:break-word;'}):
				target_url = that_span.find('b').text
				URL_DATA.append(target_url)
				#print target_url
				Data_num = Data_num+1
				
	#page2....
	for i in range(0, 9, 1):
		URL_Vonline_Page = 'https://www.phishtank.com/phish_search.php?page='+str(i)+'&valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'	
		page2 = urllib2.urlopen(URL_Vonline_Page).read()
		soup2 = BeautifulSoup(page2,"html.parser")	
		for tr2 in soup2.find_all('tr')[2:]:
		    tds2 = tr2.find_all('td')
		    P_ID2 = tds2[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
		    for i in P_ID2:
				mytxt2 = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
				link_soup2 = BeautifulSoup(mytxt2,'lxml')
				mylink2 = link_soup2.find('a')
				link_url2 = mylink2.attrs['href']
				full_url2 = base_url + link_url2
				#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
				full_resp2 = requests.get(full_url2)
				full_url_txt2 = full_resp2.text
				full_soup2 = BeautifulSoup(full_url_txt2, "lxml")
				for that_span2 in full_soup2.findAll('span',{'style':'word-wrap:break-word;'}):
					target_url2 = that_span2.find('b').text
					URL_DATA.append(target_url2)
					contrast_URL = str(URL_DATA[Data_num-1])
					if target_url2 == contrast_URL:
						check_can_save = False
						URL_DATA.pop(Data_num)
						print (("* T2 Double Data in No.%d *\n")%Data_num)
					else:
						Data_num = Data_num+1
#Return Data
	file_name = 'Unknown+Online_download'
	file = open(file_name + '.csv', 'wb')
	writer = csv.writer(file, ['Phishing_URL'])
	writer.writerow(["Phish URL"])
	for val in URL_DATA:
		writer.writerow([val])
	file.close()
	print('Saving {} '.format(file_name))

def job3_scraped_data(Valid_value,Active_value):
	#index
	Data_num = 0
	URL_DATA = []
	URL_Vonline = 'https://www.phishtank.com/phish_search.php?valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'
	page = urllib2.urlopen(URL_Vonline).read()
	soup = BeautifulSoup(page,"html.parser")	
	for tr in soup.find_all('tr')[2:]:
	    tds = tr.find_all('td')
	    P_ID = tds[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
	    for i in P_ID:
			mytxt = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
			link_soup = BeautifulSoup(mytxt,'lxml')
			mylink = link_soup.find('a')
			link_url = mylink.attrs['href']
			full_url = base_url + link_url
			#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
			full_resp = requests.get(full_url)
			full_url_txt = full_resp.text
			full_soup = BeautifulSoup(full_url_txt, "lxml")
			for that_span in full_soup.findAll('span',{'style':'word-wrap:break-word;'}):
				target_url = that_span.find('b').text
				URL_DATA.append(target_url)
				#print target_url
				Data_num = Data_num+1
				
	#page2....
	for i in range(0, 9, 1):
		URL_Vonline_Page = 'https://www.phishtank.com/phish_search.php?page='+str(i)+'&valid='+str(Valid_value)+'&active='+str(Active_value)+'&Search=Search'	
		page2 = urllib2.urlopen(URL_Vonline_Page).read()
		soup2 = BeautifulSoup(page2,"html.parser")	
		for tr2 in soup2.find_all('tr')[2:]:
		    tds2 = tr2.find_all('td')
		    P_ID2 = tds2[0]### P_ID = <td class="value" valign="center"><a href="phish_detail.php?phish_id=5595177">5595177</a></td>
		    for i in P_ID2:
				mytxt2 = str(i) #i = <a href="phish_detail.php?phish_id=5596868">5596868</a>
				link_soup2 = BeautifulSoup(mytxt2,'lxml')
				mylink2 = link_soup2.find('a')
				link_url2 = mylink2.attrs['href']
				full_url2 = base_url + link_url2
				#print full_url #<div class="url">  <b>http://www.jimsings.com/sec/app/nz.nz/index.htm</b>
				full_resp2 = requests.get(full_url2)
				full_url_txt2 = full_resp2.text
				full_soup2 = BeautifulSoup(full_url_txt2, "lxml")
				for that_span2 in full_soup2.findAll('span',{'style':'word-wrap:break-word;'}):
					target_url2 = that_span2.find('b').text
					URL_DATA.append(target_url2)
					contrast_URL = str(URL_DATA[Data_num-1])
					if target_url2 == contrast_URL:
						check_can_save = False
						URL_DATA.pop(Data_num)
						print (("* T3 Double Data in No.%d *\n")%Data_num)
					else:
						Data_num = Data_num+1
#Return Data
	file_name = 'Valid+Offline_download'
	file = open(file_name + '.csv', 'wb')
	writer = csv.writer(file, ['Phishing_URL'])
	writer.writerow(["Phish URL"])
	for val in URL_DATA:
		writer.writerow([val])
	file.close()
	print('Saving {} '.format(file_name))

def main():
	tStart = time.time()
	print("Time Start!")
	t1 = threading.Thread(target=job1_scraped_data, args=("y","y"))
	t2 = threading.Thread(target=job2_scraped_data, args=("u","y"))
	t3 = threading.Thread(target=job3_scraped_data, args=("y","n"))
	t1.start()
	t2.start()
	t3.start()
	t1.join()
	t2.join()
	t3.join()
	tEnd = time.time()
	print "It cost %f sec" % (tEnd - tStart)

if __name__ == '__main__':
	main()
