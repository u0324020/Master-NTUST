#coding by Jane
#assignment: Testing http or https and Writing CSV file
import requests
import urllib2
from time import gmtime, strftime
import csv
import timeit

start = timeit.default_timer()
URL_DATA = []
Org_URL = []
counter = 399999
#Test http or https
file = open('40wto45w.csv', 'r')
csvCursor = csv.reader(file)
for row in csvCursor:
	#time.sleep(0.5)
	counter = counter + 1
	print counter
	try:
		r = requests.get('http://'+str(row[0]), headers={'Connection':'close'},timeout=5)
		print r.url
		URL_DATA.append(r.url)
	except:
		print("except")

#write url to top-1m
file_name = 'Update_40wto45w'
file = open(file_name + '.csv', 'wb')
writer = csv.writer(file, ['Benign_URL'])
for val in URL_DATA:
	writer.writerow([val])
file.close()
print('Saving {} '.format(file_name))
stop = timeit.default_timer()
print ((stop - start)/60/60)