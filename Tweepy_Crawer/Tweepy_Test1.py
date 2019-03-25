import csv #Import csv
import tweepy 

csvFile = open('result.csv', 'a')
csvWriter = csv.writer(csvFile)
key = '39z9Oe8nXcPPLqvfXOxdWuchs'
secret ='5DyXE8BJMCEPz8yeyAhnXA6TpH2VVrTrAmyp20VnOznBnltfEu'
token = '1106489441860804608-PImjAF4NwAyePVYrqBg8GejRnQhVx5'
token_secret = 'gUfoibf4eikGirplTdvnGTdkIbQFJ8kno2SGKlDHfYoqW'

auth = tweepy.OAuthHandler(key,secret)
auth.set_access_token(token,token_secret)

api = tweepy.API(auth)
#api.update_status('Hello python')
#public_tweets = api.home_timeline()#self page
#public_tweets = api.user_timeline('LeoDiCaprio') #others page
# for tweet in public_tweets:
# 	print(tweet.text)
S_user = []
for tweet in tweepy.Cursor(api.search,q='NTUST').items(10):
	S_user.append(tweet.user.screen_name)
	print('Tweet by: @' + tweet.user.screen_name)
for i in S_user:
	public_tweets = api.user_timeline(str(i)) #others page
	for tweets in public_tweets:
		csvWriter.writerow([str(i), tweets.text.encode('utf-8')])
		print(tweets.text)
csvFile.close()