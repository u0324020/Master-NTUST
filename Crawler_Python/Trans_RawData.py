# coding by Jane -2019
# encoding: utf-8
# 把原始資料轉成RawData
# using : python 8. Tras_RawData.py <inputfile>
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


def Add_Row(row, col, data):
    if col == 1 and data == 0:
        print("Null Values")


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
    count_comments = 0
    count_comments = content.count("<!--", 0, len(content))  # html
    count_comments += content.count("/*", 0, len(content))  # js
    count_comments += content.count("//", 0, len(content))  # js_sigle
    print("!!!!!comment!!!" + str(count_comments))
    Add_Row(now_ID, 18, count_comments)
    count_links = 0
    count_links = content.count("http", 0, len(content))
    print("link = " + str(count_links))
    Add_Row(now_ID, 19, count_comments)


def Open_CSV(path):
    df = pd.read_csv(path, skipinitialspace=True)
    now_ID = 0
    for Page in df.Page:
        now_ID = now_ID + 1
        if Page == 1:
            Add_Row(now_ID, 1, 1)
            Open_txt(now_ID)
            # print("Page:" + str(now_ID))
        else:
            Add_Row(now_ID, 1, 0)
            # print("No Page:" + str(now_ID))
            # arr = "0"


if __name__ == '__main__':
    file_path = "C:/Users/Jane/Desktop/NTU/Scam/Data/" + \
        str(sys.argv[1]) + ".csv"
    Open_CSV(file_path)
    # Open_txt(6)
