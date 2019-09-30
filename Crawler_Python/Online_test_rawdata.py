# coding by Jane -2019
# encoding: utf-8
# using : python Online_test_rawdata.py <url>
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
import whois
import datetime

headers = {
    'Content-Type': 'application/json',
    #'API-Key': 'b5307850-38df-4ba8-b2e5-6b11f7a3e8ed' #1
    'API-Key': '184f3fe8-4db8-443d-99fb-17cde56ec564'  # 2
}

# task
chk_uuid = ""
scan_time = ""
org_url = ""
dom_url = ""
# page
domain = ""
country = ""
city = ""
server = ""
ip = ""
asn = ""
asn_name = ""
# stats
securePercentage = ""
IPv6Percentage = ""
uniqCountries = ""
adBlocked = ""
regDomainStats = ""
Page_size = ""
count = ""
# lists
requests_count = ""
IP_count = ""
domains_count = ""
server_count = ""
hashes_count = ""
count_time = 0
data_array = ""
ID_array = ""
Phone_array = ""
page_ERROR = ""
page = ""
page_arr = ""
content = ""

def get_url_uuid(url):
    if url != ' ':
        data = '{"url": "%s"}' % url
        data = (data.encode('utf-8'))
        response = requests.post(
            'https://urlscan.io/api/v1/scan/', headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            uuid = response.json()["uuid"]
            print("URL = %s, uuid = %s" % (url, uuid))
        else:
            print("ERROR:%s" % data)
    return uuid


def get_org(uuid,orgurl):
    url = "https://urlscan.io/result/" + uuid + "#transactions"
    page = Curl_Page(uuid, url)
    #print(page)
    page_arr = "1"
    if page == "":
        page_arr = "0"
    df = get_req(uuid,orgurl,page_arr)
    Network_based(uuid, df)
    URL_whois(uuid, df)
    if page_arr == "1":
        First_list[20] = 1
        try:
            Open_txt(uuid,page)
            print("Page:" + str(uuid))
        except:
            for i in range(20, 43):
                First_list[i] = 0
    if page_arr == "0":
        for i in range(20, 43):
            First_list[i] = 0
        print("No Page:" + str(uuid))


def Curl_Page(ID, url):
    req = str((requests.get(url)).status_code)
    print("Curl_Page: "+req)
    if "200" in req:
        print("It's 200 ok")
        try:
            soup = BeautifulSoup(urlopen(url).read(), "html.parser")
            page_arr = (1)
            return get_respense_path(ID, soup)
        except:
            page_arr = (0)
            print("page is 404")
    else:
        page_arr = (0)
    return page_arr


def get_respense_path(ID, soup):
    anchors = soup.find(
        'a', {'class': 'btn btn-xs btn-default pull-right', 'href': True, 'rel': 'nofollow'})
    hash_code = str(anchors['href'])
    url = "https://urlscan.io" + hash_code
    print("Hash URL = " + url)
    return Page_text(ID, url)


def Page_text(ID, taarget_url):
    print(taarget_url)
    r = requests.get(taarget_url)
    print(r.status_code)
    content = (r.text)
    print("####### Saving  %s.txt #######" % ID)
    #print(content)
    return content



def get_req(uuid,orgurl,page_arr):
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
            chk_uuid = (task["uuid"])
            scan_time = (task["time"])
            dom_url = (task["domURL"])
            org_url = (task["url"])
            # print(chk_uuid, scan_time, org_url, dom_url)
            domain = (page["domain"])
            country = (page["country"])
            city = (page["city"])
            server = (page["server"])
            ip = (page["ip"])
            asn = (page["asn"])
            asn_name = (page["asnname"])
            ####
            securePercentage = (stats["securePercentage"])
            IPv6Percentage = (stats["IPv6Percentage"])
            uniqCountries = (stats["uniqCountries"])
            adBlocked = (stats["adBlocked"])
            regDomainStats = (stats["regDomainStats"])
            Page_size = (reg[0]["size"])
            # lists
            requests_count = (len(lists["urls"]))
            IP_count = (len(lists["ips"]))
            domains_count = (len(lists["domains"]))
            server_count = (len(lists["servers"]))
            hashes_count = (len(lists["hashes"]))
        except:
            return Add_null(orgurl)
    else:
        return Add_null(orgurl)
    data_array = ["0", "0", orgurl, chk_uuid, scan_time, dom_url, domain, country, city, server, ip, asn,
                  asn_name, securePercentage, IPv6Percentage, uniqCountries, adBlocked,
                  Page_size, IP_count, domains_count, server_count, hashes_count, requests_count, page_arr]
    print(data_array)
    return data_array


def Add_null(orgurl):
    print("in the null")
    chk_uuid = ("0")
    scan_time = ("0")
    dom_url = ("0")
    # print(chk_uuid, scan_time, org_url, dom_url)
    org_url = orgurl
    domain = ("0")
    country = ("0")
    city = ("0")
    server = ("0")
    ip = ("0")
    asn = ("0")
    asn_name = ("0")
    ####
    securePercentage = ("0")
    IPv6Percentage = ("0")
    uniqCountries = ("0")
    adBlocked = ("0")
    regDomainStats = ("0")
    Page_size = ("0")
    # lists
    requests_count = ("0")
    IP_count = ("0")
    domains_count = ("0")
    server_count = ("0")
    hashes_count = ("0")
    data_array = ["0", "0", orgurl, chk_uuid, scan_time, dom_url, domain, country, city, server, ip, asn,
                  asn_name, securePercentage, IPv6Percentage, uniqCountries, adBlocked,
                  Page_size, IP_count, domains_count, server_count, hashes_count, requests_count, "0"]
    print(data_array)
    return data_array

def Feature_country(data):
    country_list = ["CN", "US", "EU", "TR", "RU", "TW", "BR",
                    "RO", "IN", "IT", "HU"]  # 'US,EU,TR,RU,TW,BR,RO,IN,IT,HU'
    if data == "null":
        data = "0"
    else:
        for c in country_list:
            if c in data:
                data = "1"
    if data != "1":
        data = "0"
    return data


def Network_based(ID, df):
    DNS_list = ["google.com", "youtube.com", "facebook.com", "baidu.com", "wikipedia.org",
                "yahoo.com", "qq.com", "taobao.com", "tmall.com", "twitter.com", "netflix.com"]
    ASN_list = ["16509", "203220", "32934", "15169", "11344"]
    ###### Domain ######
    domain_feature = str(df[6]).lower()
    for i in DNS_list:
        if i in domain_feature:
            domain_feature = "1"
        else:
            domain_feature = "0"
    First_list[0] = domain_feature
    ###### Country ######
    country_feature = str(df[7])
    First_list[1] = Feature_country(country_feature)
    ##### Top_City #######
    city_feature = str(df[8])
    if city_feature != "0":
        city_feature = "1"
    First_list[2] = city_feature
    ######### ASN ###########
    asn_feature = str(df[11])
    if asn_feature != "0":
        for i in ASN_list:
            if i in asn_feature:
                asn_feature = "1"
                First_list[3] = asn_feature
    else:
        First_list[3] = "0"
    ############IP###############
    First_list[4] = str(df[13])
    First_list[5] = str(df[14])
    First_list[6] = str(df[15])
    First_list[7] = str(df[16])
    if type(df[17]) == float:
        size = 0
    else:
        size = df[17]
    First_list[8] = str(round((size / 1024), 2))
    First_list[9] = str(df[18])
    First_list[10] = str(df[19])
    First_list[11] = str(df[20])
    First_list[12] = str(df[21])
    First_list[13] = str(df[22])


def URL_whois(ID, df):
    URL = str(df[2])
    samename_domain = 0
    samename_email = 0
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    now = datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
    try:
        whois_detals = whois.whois(URL)
        try:
            update = datetime.datetime.strptime(
                str(whois_detals['updated_date'][0]), "%Y-%m-%d %H:%M:%S")
        except:
            update = 0
        try:
            creation = datetime.datetime.strptime(
                str(whois_detals['creation_date']), "%Y-%m-%d %H:%M:%S")
        except:
            creation = 0
        try:
            expiration = datetime.datetime.strptime(
                str(whois_detals['expiration_date'][0]), "%Y-%m-%d %H:%M:%S")
        except:
            expiration = 0
        if type(update) == type(now):
            de_update = (now - update).total_seconds()
        else:
            de_update = 0
        if type(creation) == type(now):
            de_creation = (now - creation).total_seconds()
        else:
            de_creation = 0
        if type(de_creation) == type(now):
            de_once_update = de_creation - de_update
        else:
            de_once_update = 0
        if type(expiration) == type(now):
            de_expiration = (expiration - now).total_seconds()
        else:
            de_expiration = 0
        try:
            count_DNS = len(whois_detals['name_servers'])
        except:
            count_DNS = 0
        try:
            domain = whois_detals['domain_name']
        except:
            domain = 0
        try:
            email = whois_detals['emails']
        except:
            email = 0
        if domain or email != 0:
            try:
                org = whois_detals['org'].lower().replace(
                    ",", "").replace(".", "").replace("inc", "").lstrip()
                for i in domain:
                    if org in i.lower():
                        samename_domain += 1
                for i in email:
                    if org in i.lower():
                        samename_domain += 1
            except:
                org = 0
    except:
        samename_domain = 0
        count_DNS = 0
        de_creation = 0
        de_update = 0
        de_expiration = 0
        de_once_update = 0

    First_list[14] = samename_domain
    First_list[15] = count_DNS
    First_list[16] = round((de_creation / 100000), 2)
    First_list[17] = round((de_update / 100000), 2)
    First_list[18] = round((de_expiration / 100000), 2)
    First_list[19] = round((de_once_update / 100000), 2)


def Open_txt(now_ID,txt):
    content = (txt.lower())
    # print(content)
    count_keywords = 0
    brands = ["microsoft", "apple", "dell", "yahoo", "hp"]
    for i in brands:
        if i.lower() in content:
            print(i)
            First_list[21] = 1
            count_keywords = count_keywords + 1
        else:
            First_list[21] = 0

    keywords = ["support", "phone", "contact", "free", "protect", "update", "window", "alert", "document", "eval", "var",
                "window.alert", "window.confirm", "window.open", "fullscreen", "window.onload",
                "window.location", "window.AudioContext", "window.onbeforeunload"]
    location = 22
    for i in keywords:
        if i.lower() in content:
            print(i)
            First_list[location] = 1
            count_keywords = count_keywords + 1
        else:
            First_list[location] = 0
        location = location + 1
    print("###############Count = " + str(count_keywords))
    First_list[41] = count_keywords
    count_comments = 0
    count_comments = content.count("<!--", 0, len(content))  # html
    count_comments += content.count("/*", 0, len(content))  # js
    count_comments += content.count("//", 0, len(content))  # js_sigle
    print("!!!!!comment!!!" + str(count_comments))
    First_list[42] = count_comments
    count_links = 0
    count_links = content.count("http", 0, len(content))
    print("link = " + str(count_links))
    First_list[43] = count_links

if __name__ == '__main__':
    tStart = time.time()
    First_list = [0] * 45
    url = str(sys.argv[1])
    # uuid
    uuid = get_url_uuid(url)
    time.sleep(60)
    # get org
    #uuid = "044006df-e73e-42c1-b1a9-ab7b1a006086"
    get_org(uuid,url)
    print(First_list)
    tEnd = time.time()
    print ("It cost %f sec" % (tEnd - tStart))
