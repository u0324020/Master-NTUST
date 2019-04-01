# -*- coding: utf-8 -*-
#API resorce : http://docs.tweepy.org/en/v3.5.0/api.html
import csv 
import time
import json
import tweepy
import sys
from tweepy import OAuthHandler
from pandas import pandas as pd

def get_User_json(Target_User):
	user_handler = Target_User
	status_list = api.user_timeline(user_handler)
	status = status_list[0]
	json_str = json.dumps(status._json)
	#print(json_str)#json
	data = json.loads(json_str)
	id=data['id_str']
	get_tweet_reploes(user_handler,id)

def get_tweet_reploes(Target_User,id_str):
	tweet_arr=[]
	replies=[]
	user_name = str(Target_User)
	non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
	Replies_Count = 0
	now_time = time.strftime("%Y%m%d")
	for full_tweets in tweepy.Cursor(api.user_timeline,screen_name=user_name,timeout=999999).items(10):
		now_tweet = full_tweets.text.translate(non_bmp_map) #tweet
		tweet_arr.append(now_tweet)
		if (now_tweet!=" "):
			Replies_Count+=1
			#tweet_arr.append(now_tweet)
			for tweet in tweepy.Cursor(api.search,q='to:'+user_name, since_id=id_str, result_type='recent',timeout=999999).items(1000):
				if hasattr(tweet, 'in_reply_to_status_id_str'):
					if (tweet.in_reply_to_status_id_str==full_tweets.id_str):
						replies.append(tweet.text) #many_replies
			print(replies)
			if replies:
				df = pd.DataFrame({'Tweet_ID_Number':id_str+'_'+str(Replies_Count),'Reply':replies}) #secret key ID
				file_name = id_str+'_'+str(Replies_Count)+'.csv'
				df.to_csv(file_name, index = None,header=None,encoding='utf_8_sig')
				print('########Saving {} '.format(file_name))
			replies.clear()
			title_file_name = id_str+'_'+user_name+'.csv'
	csvFile = open(title_file_name, 'a',newline='')
	csvWriter = csv.writer(csvFile)
	#df_tw = pd.DataFrame({'Tweet_Title_ID':MD5_user+str(ID_number),'Tweet_content':tweet_arr})
	csvWriter.writerow(["Tweet_ID", "Tweet_content"])
	print(tweet_arr)
	for ID_number in range(0,Replies_Count):
		#print(tweet_arr[ID_number])
		csvWriter.writerow([id_str+"_"+str(ID_number+1), tweet_arr[ID_number]])
	csvFile.close()
	print('########Saving {} '.format(title_file_name))
	

def get_targetUser():
	S_user = []
	S_replies = []
	#抓出與NTUST相關文章的帳號
	for tweet in tweepy.Cursor(api.search,q='NTUST',since_id=1111163636800602113).items(10):
		S_user.append(tweet.user.screen_name)
		S_replies.append(tweet.text)
		print('Tweet by: @' + tweet.user.screen_name)

def get_timeline():
	csvFile = open('result_2.csv', 'a')
	csvWriter = csv.writer(csvFile)
	public_tweets = api.user_timeline('briian',since_id='2019-03-20',page=1,count=20, full_text=True) #others page
	for tweets in public_tweets:
		csvWriter.writerow(['briian', tweets.created_at, tweets.text.encode('utf-8')])
		print(tweets.text)
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

def get_hashtage():
	csvWriter = csv.writer(csvFile)
	for tweet in tweepy.Cursor(api.search,q="#ps4",count=100,lang="en",	since_id='2019-03-12').items():
		print(tweet.created_at, tweet.text)
		csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

if __name__ == '__main__':

	auth = tweepy.OAuthHandler(key,secret)
	auth.set_access_token(token,token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True)
	Target_User = 'realDonaldTrump'
	get_User_json(Target_User)
	# for tweet in tweepy.Cursor(api.search,q='NTUST').items(10):
	# 	print('Tweet by: @' + tweet.user.screen_name)
	# 	print(tweet)