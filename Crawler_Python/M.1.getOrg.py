# coding by Jane -2019
# encoding: utf-8
# 把uuid相對應的資料JSON抓下來<更新>
# using : python M.1.getOrg.py <inputfile> <outputfile> <startRow>
import requests
import time
from io import BytesIO
from base64 import b64encode
import pandas as pd
import numpy as np
import csv
from multiprocessing import Queue
from bs4 import BeautifulSoup
import threading
import requests
from urllib.request import urlopen
import re
import sys

# task
chk_uuid = []
scan_time = []
org_url = []
dom_url = []
# page
domain = []
country = []
city = []
server = []
ip = []
asn = []
asn_name = []
# stats
securePercentage = []
IPv6Percentage = []
uniqCountries = []
adBlocked = []
regDomainStats = []
Page_size = []
count = []
# lists
requests_count = []
IP_count = []
domains_count = []
server_count = []
hashes_count = []
count_time = 0
data_array = []
ID_array = []
Phone_array = []
page_ERROR = []
page = []


def Curl_Page(ID, url):
    req = str((requests.get(url)).status_code)
    if "200" in req:
        print("It's 200 ok")
        try:
            soup = BeautifulSoup(urlopen(url).read(), "html.parser")
            return get_respense_path(ID, soup)
        except:
            Epath = "C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/txt_ERROR.txt"
            with open(Epath, 'a', encoding='utf-8') as fE:
                fE.write("#%s URL = %s\n" % (ID, url))
                fE.close()
            print("It's 404")
    else:
        Epath = "C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/txt_ERROR.txt"
        with open(Epath, 'a', encoding='utf-8') as fE:
            fE.write("#%s URL = %s\n" % (ID, url))
            fE.close()


def get_respense_path(ID, soup):
    # role="tabpanel" class="main-pane" id="iocs"
    # print("get_respense_path")
    count = 0
    anchors = soup.find(
        'a', {'class': 'btn btn-xs btn-default pull-right', 'href': True, 'rel': 'nofollow'})
    hash_code = str(anchors['href'])
    url = "https://urlscan.io" + hash_code
    print("Hash URL = " + url)
    Page_text(ID, url)


def Page_text(ID, taarget_url):
    print("taarget_url")
    print(taarget_url)
    r = requests.get(taarget_url)
    print(r.status_code)
    if str(r.status_code) == "200":
        path = "C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/M_txt/" + \
            str(ID) + ".txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(r.text)
            f.close()
        print("####### Saving  %d.txt #######" % ID)
    else:
        print("2:404")
        Epath = "C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/txt_ERROR.txt"
        with open(Epath, 'a', encoding='utf-8') as fE:
            fE.write("#%s URL = %s\n" % (ID, url))
            fE.close()
        #Saving_txt(ID, (str(r.status_code)))


def get_req(uuid):
    print("get_req:" + str(uuid))
    url = "https://urlscan.io/api/v1/result/" + uuid
    payload = {
        "url": url,
        "content_type": "json",
        "method": "get",
        "expected_update_period_in_days": "1"}
    req = requests.get(
        url, params=payload)
    if (req.status_code) == 200:
        try:
            req_json = req.json()
            task = req_json["task"]  # uuid, time, url, domURL
            # domain, country, city, server, ip, asn, asnname
            page = req_json["page"]
            # securePercentage, IPv6Percentage, uniqCountries, adBlocked, regDomainStats
            stats = req_json["stats"]
            lists = req_json["lists"]
            reg = stats["regDomainStats"]
            chk_uuid .append(task["uuid"])
            scan_time .append(task["time"])
            dom_url .append(task["domURL"])
            org_url.append(task["url"])
            # print(chk_uuid, scan_time, org_url, dom_url)
            domain .append(page["domain"])
            country .append(page["country"])
            city .append(page["city"])
            server .append(page["server"])
            ip .append(page["ip"])
            asn .append(page["asn"])
            asn_name .append(page["asnname"])
            ####
            securePercentage .append(stats["securePercentage"])
            IPv6Percentage .append(stats["IPv6Percentage"])
            uniqCountries .append(stats["uniqCountries"])
            adBlocked .append(stats["adBlocked"])
            regDomainStats .append(stats["regDomainStats"])
            Page_size .append(reg[0]["size"])
            # lists
            requests_count .append(len(lists["urls"]))
            IP_count .append(len(lists["ips"]))
            domains_count .append(len(lists["domains"]))
            server_count .append(len(lists["servers"]))
            hashes_count .append(len(lists["hashes"]))
        except:
            Add_null()
    else:
        Add_null()


def Add_null():
    chk_uuid .append("0")
    scan_time .append("0")
    dom_url .append("0")
    # print(chk_uuid, scan_time, org_url, dom_url)
    org_url.append("0")
    domain .append("0")
    country .append("0")
    city .append("0")
    server .append("0")
    ip .append("0")
    asn .append("0")
    asn_name .append("0")
    ####
    securePercentage .append("0")
    IPv6Percentage .append("0")
    uniqCountries .append("0")
    adBlocked .append("0")
    regDomainStats .append("0")
    Page_size .append("0")
    # lists
    requests_count .append("0")
    IP_count .append("0")
    domains_count .append("0")
    server_count .append("0")
    hashes_count .append("0")


def DF_Table(count_array, ID, Phone):
    n = count_array
    ID_array.append(ID)
    Phone_array.append(Phone)
    data_array = [ID_array[n], Phone_array[n], org_url[n], chk_uuid[n], scan_time[n], dom_url[n], domain[n], country[n], city[n], server[n], ip[n], asn[n],
                  asn_name[n], securePercentage[n], IPv6Percentage[n], uniqCountries[n], adBlocked[n],
                  Page_size[n], IP_count[n], domains_count[n], server_count[n], hashes_count[n], requests_count[n], "1"]
    df = pd.DataFrame(data=[data_array], index=None,
                      columns=None, dtype=None, copy=False)
    # df.loc[org_url, 'url'] = re.sub(r"[\n\t\s]*", "", org_url)
    df.to_csv("C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/Malicious_" + str(sys.argv[2]) + ".csv", encoding="utf-8",
              mode='a', index=0, sep=',', header=None)
    # df2 = pd.read_csv("urlscan_test.csv", encoding="utf-8")
    # print(df2)


def Write_head():
    head = ["ID", "Phone", "url", "uuid", "time", "domURL", "domain", "country", "city", "server", "ip", "asn",
            "asnname", "securePercentage", "IPv6Percentage", "uniqCountries", "adBlocked",
            "page_size_KB", "IP_count", "domains_count", "server_count", "hashes_count", "requests_count", "Page"]
    df = pd.DataFrame(data=None, index=None,
                      columns=head, dtype=None, copy=False)
    df.to_csv("C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/Malicious_" + str(sys.argv[2]) + ".csv", encoding="utf-8",
              mode='w', index=0, sep=',')
    return df


def Open_File(filename):
    df = pd.read_csv(filename + '.csv', skipinitialspace=True)
    count_time = int(sys.argv[3])
    count_array = 0
    for uuid in df.uuid[count_time:]:
        count_time = count_time + 1
        print("############  %d  ###########" % count_time)
        url = "https://urlscan.io/result/" + uuid + "#transactions"
        Curl_Page(count_time, url)
        try:
            get_req(uuid)
        except:
            f = open('C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/ERROR.txt', 'a')
            f.write("ID = " + str(count_time) +
                    ", uuid = " + str(uuid) + "\n")
            print("ERROR:%s" % uuid)
            pass
        DF_Table(count_array, count_time,
                 0)
        count_array = count_array + 1
        print('Saving ID = {} , uuid = {}'.format(count_time, uuid))


if __name__ == '__main__':
    file_path = "C:/Users/Jane/Desktop/NTU/Scam/Data/Malicious/Malicious_" + \
        str(sys.argv[1])
    Save_DF = Write_head()
    Open_File(file_path)
    # now = time_S - (time.gmtime())
    # print(str(now))
