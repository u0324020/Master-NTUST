# coding by Jane -2019
# encoding: utf-8
# 把原始資料轉成RawData
# using : python 8. get_urlscan_page.py <inputfile>
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

# domain_feature, country_feature, city_feature, asn_feature, securePercentage_feature, IPv6Percentage_feature,
# uniqCountries_feature, adBlocked_feature, page_size_KB_feature, IP_count_feature, domains_count_feature, server_count_feature,
# hashes_count_feature, requests_count_feature, samename_domain, count_DNS, de_creation, de_update, de_expiration, de_once_update = ""


def Add_CSV(arr):
    df = pd.DataFrame(data=[arr], index=None,
                      columns=None, dtype=None, copy=False)
    df.to_csv("C:/Users/Jane/Desktop/NTU/Scam/Data/" + str(sys.argv[2]) + ".csv", encoding="utf-8",
              mode='a', index=0, sep=',', header=None)


def Feature_country(data):
    country_list = []  # 'US,EU,TR,RU,TW,BR,RO,IN,IT,HU'
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
    ID = ID - 1
    ###### Domain ######
    domain_feature = str(df.domain[ID]).lower()
    for i in DNS_list:
        if i in domain_feature:
            domain_feature = "1"
        else:
            domain_feature = "0"
    First_list[0] = domain_feature
    ###### Country ######
    country_feature = str(df.country[ID])
    First_list[1] = Feature_country(country_feature)
    ##### Top_City #######
    city_feature = str(df.city[ID])
    if city_feature != "0":
        city_feature = "1"
    First_list[2] = city_feature
    ######### ASN ###########
    asn_feature = str(df.asn[ID])
    if asn_feature != "0":
        for i in ASN_list:
            if i in asn_feature:
                asn_feature = "1"
                First_list[3] = asn_feature
    else:
        First_list[3] = "0"
    ############IP###############
    First_list[4] = str(df.securePercentage[ID])
    First_list[5] = str(df.IPv6Percentage[ID])
    First_list[6] = str(df.uniqCountries[ID])
    First_list[7] = str(df.adBlocked[ID])
    First_list[8] = str(df.page_size_KB[ID])
    First_list[9] = str(df.IP_count[ID])
    First_list[10] = str(df.domains_count[ID])
    First_list[11] = str(df.server_count[ID])
    First_list[12] = str(df.hashes_count[ID])
    First_list[13] = str(df.requests_count[ID])


def URL_whois(ID, df):
    ID = ID - 1
    URL = str(df.url[ID])
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
    First_list[16] = de_creation
    First_list[17] = de_update
    First_list[18] = de_expiration
    First_list[19] = de_once_update


def Open_txt(now_ID):
    txt = open("C:/Users/Jane/Desktop/NTU/Scam/Data/txt - final/" +
               str(now_ID) + ".txt", "r")
    content = (txt.read()).lower()
    # print(content)
    count_keywords = 0
    brands = []
    for i in brands:
        if i.lower() in content:
            print(i)
            First_list[21] = 1
            count_keywords = count_keywords + 1
        else:
            First_list[21] = 0

    keywords = []
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


def Open_CSV(path):
    df = pd.read_csv(path, skipinitialspace=True)
    now_ID = 0
    for Page in df.Page:
        now_ID = now_ID + 1
        Network_based(now_ID, df)
        URL_whois(now_ID, df)
        print("For Page = " + str(now_ID))
        if str(Page) == "1":
            First_list[20] = 1
            Open_txt(now_ID)
            print("Page:" + str(now_ID))
        if str(Page) == "0":
            for i in range(20, 43):
                First_list[i] = 0
            print("No Page:" + str(now_ID))
        # print(Final_list[:now_ID][:23])
        print(First_list[:43])
        Add_CSV(First_list)


if __name__ == '__main__':
    file_path = "C:/Users/Jane/Desktop/NTU/Scam/Data/" + \
        str(sys.argv[1]) + ".csv"
    First_list = [0] * 45
    Open_CSV(file_path)
