# -*- coding: utf-8 -*-
#get json
import csv 
import time
import json
import tweepy
from tweepy import OAuthHandler


def get_targetUser():
	S_user = []
	#抓出與NTUST相關文章的帳號
	for tweet in tweepy.Cursor(api.search,q='NTUST').items(10):
		S_user.append(tweet.user.screen_name)
		print('Tweet by: @' + tweet.user.screen_name)

def get_timeline():
	csvFile = open('result_1.csv', 'a')
	csvWriter = csv.writer(csvFile)
	for i in S_user:
		public_tweets = api.user_timeline(str(i),page=1,count=200, full_text=True) #others page
		for tweets in public_tweets:
			csvWriter.writerow([str(i), tweets.created_at, tweets.text.encode('utf-8')])
			#print(tweets.text)
	csvFile.close()
def get_FollowerID():
	#McDonalds帳號的Follower ID
	ids = []
	for page in tweepy.Cursor(api.followers_ids, screen_name="J_ARamsey").pages():
		ids.extend(page)
		print(page)
		time.sleep(60)
	print(len(ids))

def get_FollowerName():
	#ikhsan_chemy帳號的Follower Name
	for follower in api.followers_ids('ikhsan_chemy'):
		print(api.get_user(follower).screen_name)

def get_User_json():
	user_handler = 'ikhsan_chemy'
	status_list = api.user_timeline(user_handler)
	status = status_list[0]
	json_str = json.dumps(status._json)
	#print(json_str)#json
	data = json.loads(json_str)
	print(data['place']['country_code'])#

def get_hashtage():
	csvWriter = csv.writer(csvFile)
	for tweet in tweepy.Cursor(api.search,q="#ps4",count=100,\
	lang="en",\
	since_id='2019-03-12').items():
		print(tweet.created_at, tweet.text)
		csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

if __name__ == '__main__':
	key = '39z9Oe8nXcPPLqvfXOxdWuchs'
	secret ='5DyXE8BJMCEPz8yeyAhnXA6TpH2VVrTrAmyp20VnOznBnltfEu'
	token = '1106489441860804608-PImjAF4NwAyePVYrqBg8GejRnQhVx5'
	token_secret = 'gUfoibf4eikGirplTdvnGTdkIbQFJ8kno2SGKlDHfYoqW'
	auth = tweepy.OAuthHandler(key,secret)
	auth.set_access_token(token,token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True)
	# for tweet in tweepy.Cursor(api.search,q='NTUST').items(10):
	# 	print('Tweet by: @' + tweet.user.screen_name)
	# 	print(tweet)