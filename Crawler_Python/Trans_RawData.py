# coding by Jane -2019
# encoding: utf-8
# 把原始資料轉成RawData
# using : python 8. get_urlscan_page.py <inputfile> <startRow>
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


def Add_Row(row, col, data):
    if col == 1 and data == 0:
        print("None")


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
    print("country" + data)


def Network_based(ID, df):
    ID = ID

    DNS_list = []
    ASN_list = []
    ####### Domain ######
    # domain_feature = str(df.domain[ID]).lower()
    # for i in DNS_list:
    #     if i in domain_feature:
    #         domain_feature = "1"
    #     else:
    #         domain_feature = "0"
    # print(domain_feature)
    ####### Country ######
    #country_feature = str(df.country[ID])
    #country_feature = Feature_country(country_feature)
    ###### Top_City #######
    # city_feature = str(df.city[ID])
    # if city_feature != "0":
    #     city_feature = "1"
    # print("city = " + city_feature)
    ########## ASN ###########
    # asn_feature = str(df.asn[ID])
    # if asn_feature != "0":
    #     for i in ASN_list:
    #         if i in asn_feature:
    #             asn_feature = "1"
    #             print("asn = " + asn_feature)
    #############IP###############
    securePercentage_feature = str(df.securePercentage[ID])
    IPv6Percentage_feature = str(df.IPv6Percentage[ID])
    uniqCountries_feature = str(df.uniqCountries[ID])
    adBlocked_feature = str(df.adBlocked[ID])
    page_size_KB_feature = str(df.page_size_KB[ID])
    IP_count_feature = str(df.IP_count[ID])
    domains_count_feature = str(df.domains_count[ID])
    server_count_feature = str(df.server_count[ID])
    hashes_count_feature = str(df.hashes_count[ID])
    requests_count_feature = str(df.requests_count[ID])


def URL_whois(URL):
    samename_domain = 0
    samename_email = 0
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    now = datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
    whois_detals = whois.whois(URL)
    update = datetime.datetime.strptime(
        str(whois_detals['updated_date'][0]), "%Y-%m-%d %H:%M:%S")
    creation = datetime.datetime.strptime(
        str(whois_detals['creation_date']), "%Y-%m-%d %H:%M:%S")
    expiration = datetime.datetime.strptime(
        str(whois_detals['expiration_date'][0]), "%Y-%m-%d %H:%M:%S")

    de_update = (now - update).total_seconds()
    de_creation = (now - creation).total_seconds()
    de_onec_update = de_creation - de_update
    de_expiration = (expiration - now).total_seconds()
    count_DNS = len(whois_detals['name_servers'])
    org = whois_detals['org'].lower().replace(
        ",", "").replace(".", "").replace("inc", "").lstrip()
    domain = whois_detals['domain_name']
    email = whois_detals['emails']

    for i in domain:
        if org in i.lower():
            samename_domain += 1
        else:
            print("no_samename_domain")
    for i in email:
        if org in i.lower():
            samename_email += 1
        else:
            print("no_samename_email")
    print(de_update)
    print(de_creation)
    print(de_expiration)
    print(de_onec_update)
    print(count_DNS)
    print(samename_domain)
    print(samename_email)


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
            Add_Row(now_ID, 2, 1)
            count_keywords = count_keywords + 1
        else:
            Add_Row(now_ID, 2, 0)

    keywords = []
    location = 0
    for i in keywords:
        if i.lower() in content:
            print(i)
            Add_Row(now_ID, location, 1)
            count_keywords = count_keywords + 1
        else:
            Add_Row(now_ID, location, 0)
        location = location + 1
    print("###############Count = " + str(count_keywords))
    Add_Row(now_ID, 21, str(count_keywords))
    count_comments = 0
    count_comments = content.count("<!--", 0, len(content))  # html
    count_comments += content.count("/*", 0, len(content))  # js
    count_comments += content.count("//", 0, len(content))  # js_sigle
    print("!!!!!comment!!!" + str(count_comments))
    Add_Row(now_ID, 22, count_comments)
    count_links = 0
    count_links = content.count("http", 0, len(content))
    print("link = " + str(count_links))
    Add_Row(now_ID, 23, count_comments)


def Open_CSV(path):
    df = pd.read_csv(path, skipinitialspace=True)
    now_ID = 0
    for Page in df.Page:
        if Page == 1:
            Add_Row(now_ID, 1, 1)
            # Open_txt(now_ID)
            Network_based(now_ID, df)
            # print("Page:" + str(now_ID))
        else:
            Add_Row(now_ID, 1, 0)
            # print("No Page:" + str(now_ID))
            # arr = "0"
        now_ID = now_ID + 1


if __name__ == '__main__':
    file_path = "C:/Users/Jane/Desktop/NTU/Scam/Data/" + \
        str(sys.argv[1]) + ".csv"
    Open_CSV(file_path)
    # Open_txt(6)
