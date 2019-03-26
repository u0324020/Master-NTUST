# -*- coding: utf-8 -*-
import csv #Import csv
import tweepy 
import time

csvFile = open('result.csv', 'a')
csvWriter = csv.writer(csvFile)
key = '39z9Oe8nXcPPLqvfXOxdWuchs'
secret ='5DyXE8BJMCEPz8yeyAhnXA6TpH2VVrTrAmyp20VnOznBnltfEu'
token = '1106489441860804608-PImjAF4NwAyePVYrqBg8GejRnQhVx5'
token_secret = 'gUfoibf4eikGirplTdvnGTdkIbQFJ8kno2SGKlDHfYoqW'

auth = tweepy.OAuthHandler(key,secret)
auth.set_access_token(token,token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)
#api.update_status('Hello python')
#public_tweets = api.home_timeline()#self page
#public_tweets = api.user_timeline('LeoDiCaprio') #others page
# for tweet in public_tweets:
# 	print(tweet.text)
S_user = []
#抓出與NTUST相關文章的帳號
for tweet in tweepy.Cursor(api.search,q='NTUST').items(10):
	S_user.append(tweet.user.screen_name)
	print('Tweet by: @' + tweet.user.screen_name)
#McDonalds帳號的Follower
ids = []
for page in tweepy.Cursor(api.followers_ids, screen_name="J_ARamsey").pages():
	ids.extend(page)
	print(page)
	time.sleep(60)
print(len(ids))

#依帳號抓所有資訊存入CSV
for i in S_user:
	public_tweets = api.user_timeline(str(i),page=1,count=200, full_text=True) #others page
	for tweets in public_tweets:
		csvWriter.writerow([str(i), tweets.created_at, tweets.text.encode('utf-8')])
		#print(tweets.text)
csvFile.close()