'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
TODO
*****

'''
import tweepy

class TweetStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)


streamer = TweetStreamListener()

key = list(open('keys.txt'))
for k in range(len(key)):
	key[k] = key[k].strip()

auth = tweepy.OAuthHandler(key[0], key[1])
auth.set_access_token(key[2], key[3])

api = tweepy.API(auth)

public_tweets = api.home_timeline()
# for tweet in public_tweets:
    # print(tweet.text)

myStream = tweepy.Stream(auth = api.auth, listener=streamer)
myStream.filter(track=['python'])