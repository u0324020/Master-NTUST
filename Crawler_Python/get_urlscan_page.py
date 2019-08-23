# coding by Jane -2019
# encoding: utf-8
# 把uuid相對應的資料Page抓下來
# using : python get_urlscan_page.py <inputfile>
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

page_contents = []


def Curl_Page(ID, url):
    req = str((requests.get(url)).status_code)
    if "200" in req:
        print("It's 200 ok")
        try:
            soup = BeautifulSoup(urlopen(url).read(), "html.parser")
            page_contents.append(req)
            return get_respense_path(ID, soup)
        except:
            print("It's 404")
            page_contents.append(req)
    else:
        print("It's 404")
        page_contents.append(req)


def get_respense_path(ID, soup):
    # role="tabpanel" class="main-pane" id="iocs"
    print("get_respense_path")
    count = 0
    for i in soup.find_all('div', {'role': 'tabpanel', 'class': 'main-pane', 'id': 'iocs'}):
        # class="col col-md-12"
        for j in i.find_all(href=True):
            count = count + 1
            if count == 4:
                hash_code = str(j.renderContents()).replace(
                    "b'", "", 1).replace("'", "")  # 找response hash code
    url = "https://urlscan.io/responses/" + hash_code + "/"
    Page_text(ID, url)


def Page_text(ID, taarget_url):
    print("taarget_url")
    print(taarget_url)
    r = requests.get(taarget_url)
    print(r.status_code)
    if str(r.status_code) == "200":
        print("to saving")
        Saving_txt(ID, r.text)
    else:
        Saving_txt(ID, (str(r.status_code)))


def Open_File(filename):
    df = pd.read_csv(filename + '.csv', skipinitialspace=True)
    count = 0
    for uuid in df.uuid:
        count = count + 1
        print("ID:" + str(count) + "#######" + uuid + "########")
        url = "https://urlscan.io/result/" + uuid + "#transactions"
        Curl_Page(count, url)
    path = "C:/Users/Jane/Desktop/NTU/Scam/Data/txt/Report.txt"
    file2 = open(path, 'w')
    file2.write(str(page_contents))
    file2.close()
    print("######### Saving Report #######")


# def Insert_csv(page_contents):
#     df2 = pd.read_csv(
#         "C:/Users/Jane/Desktop/NTU/Scam/Data/" + str(sys.argv[2]) + ".csv")
#     df2.insert(23, "content", page_contents)
#     print("############  Insert  ###########")

def Saving_txt(ID, contents):
    path = "C:/Users/Jane/Desktop/NTU/Scam/Data/txt/" + str(ID) + ".txt"
    file = open(path, 'w')
    # print(page_contents)
    # print(type(page_contents))
    file.write(str(contents))
    file.close()
    print("### saving ####")


if __name__ == '__main__':
    file_path = "C:/Users/Jane/Desktop/NTU/Scam/Data/" + str(sys.argv[1])
    Open_File(file_path)
