import tweepy 

key = '39z9Oe8nXcPPLqvfXOxdWuchs'
secret ='5DyXE8BJMCEPz8yeyAhnXA6TpH2VVrTrAmyp20VnOznBnltfEu'
token = '1106489441860804608-PImjAF4NwAyePVYrqBg8GejRnQhVx5'
token_secret = 'gUfoibf4eikGirplTdvnGTdkIbQFJ8kno2SGKlDHfYoqW'

auth = tweepy.OAuthHandler(key,secret)
auth.set_access_token(token,token_secret)

api = tweepy.API(auth)
