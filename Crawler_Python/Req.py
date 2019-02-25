# coding by Jane -2018
# encoding: utf-8
import requests
import time
from io import BytesIO
from base64 import b64encode
'''
headers = {
            'Content-Type': 'application/json',
            'API-Key': 'b5307850-38df-4ba8-b2e5-6b11f7a3e8ed',
        }
target_urls = "http://shenandy184.pixnet.net/blog"

data = '{"url": "%s"}' % target_urls
response = requests.post('https://urlscan.io/api/v1/scan/', headers=headers, data=data)

if response.status_code == 200:
    uuid = response.json()["uuid"]
print uuid
time.sleep(30)
grab_report(uuid)'''
uuid = 'e9af8b6b-d773-48d9-96f0-ce6d1fbbc106'

response = requests.get("https://urlscan.io/api/v1/result/%s" % uuid)

if response.status_code == 200:
    screenshot = BytesIO(requests.get("https://urlscan.io/screenshots/%s.png" % uuid).content)
    img_base64 = (b64encode(screenshot.getvalue())).decode("utf-8")

    dom = requests.get("https://urlscan.io/dom/%s/" % uuid).content.decode("utf-8")

    request_response_chain = {}

    request_index = 0
    for request in response.json()['data']['requests']:
        request_response_chain[request_index] = {}
        if 'request' in request['request']:
            request_response_chain[request_index]["mime_type"] = request['request']['type'] if 'type' in request['request'] else ''
            request_response_chain[request_index]["method"] = request['request']['request']['method'] if 'method' in request['request']['request'] else ''
            request_response_chain[request_index]["request_url"] = request['request']['request']['url'] if 'url' in request['request']['request'] is not '' else ''
        else:
            request_response_chain[request_index]["mime_type"] = ''
            request_response_chain[request_index]["method"] = ''
            request_response_chain[request_index]["request_url"] = ''


        if 'response' in request:
            if 'response' in request['response']:
                request_response_chain[request_index]["response_status"] = request['response']['response']['status'] if 'status' in  request['response']['response'] else ''
                request_response_chain[request_index]["response_IP_PORT"] = "{0}:{1}".format(request['response']['response']['remoteIPAddress'], request['response']['response']['remotePort']) if 'remoteIPAddress' in request['response']['response'] is not '' else ''
                request_response_chain[request_index]["ip_whois_name"] = request['response']['asn']['name'] if 'asn' in request['response'] else ''
        else:
            request_response_chain[request_index]["response_status"] = ''
            request_response_chain[request_index]["response_IP_PORT"] = ''
            request_response_chain[request_index]["ip_whois_name"] = ''

        if 'requests' in request:
            redirects = []
            for inner_request in request['requests']:
                redirects.extend([inner_request['request']['url']])
            request_response_chain[request_index]["redirects"] = redirects
        else:
            if 'redirectResponse' in request['request']:
                redirects = []
                redirects.insert(0, request['request']['redirectResponse']['url'])
                redirects.insert(1, request['request']['request']['url'])
                request_response_chain[request_index]["redirects"] = redirects


        request_index += 1

    if 'redirects' in request_response_chain[0]:
        effective_url = request_response_chain[0]['redirects'][-1]
    else:
        effective_url = ''

    formatted_response = {
        "ips" : response.json()['lists']['ips'],
        "countries" : response.json()['lists']['countries'],
        "domains" : response.json()['lists']['domains'],
        "urls" : response.json()['lists']['urls'],
        "linkDomains" : response.json()['lists']['linkDomains'],
        "certificates" : response.json()['lists']['certificates'],
        "ans": response.json()['lists']['asns'],
        "page" : response.json()['page'],
        "task" : response.json()['task'],
        "malicious" : response.json()['stats']['malicious'],
        "adBlocked" : response.json()['stats']['adBlocked'],
        "effective_url" : effective_url,
        "request_response_chain" : request_response_chain
    }


    build_report = {"urlscan_response": formatted_response,
                    "urlscan_screenshot": img_base64,
                    "urlscan_dom" : dom
                    }

    


#curl https://urlscan.io/api/v1/result/
'''
def main():
    initialize()
    if hasattr(args, 'url'):
        submit()

    if hasattr(args, 'host'):
        search()

    if hasattr(args, 'uuid'):
        query()'''