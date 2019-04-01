import tweepy
from tweepy import OAuthHandler
import time
import json
import sys
import os
import csv
import pandas as pd
from collections import OrderedDict

consumer_key = 'key'
consumer_secret = 'secret'
access_token = 'token'
access_secret = 'access secret'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

with open (os.path.expanduser("users.csv")) as csvfile: #open file containing list of Twitter screennames
    users = csvfile.readlines()

users_processed = list(OrderedDict.fromkeys(users)) #re-order and filter duplicates
users_and_tweets = [] 

while True: #loop through list of users and obtain timestamps of most recent 200 tweets
    try:
        for user in users_processed:
            tweets = api.user_timeline(screen_name = user, count = 200, include_rts = True)
            for t in tweets:
                users_and_tweets.append({'User': user, 'Time': t.created_at})
    except tweepy.TweepError:
        time.sleep(60*15)
        continue

    except IOError:
        time.sleep(60*5)
        continue

    except StopIteration:
        break

        
df = pd.DataFrame(users_and_tweets) #turn list of dictionaries into dataframe
df_pivoted = df.pivot(column = "User", values = "Time") #pivot to separate tweet frequencies by username
df_pivoted.to_csv('users_and_tweets.csv')