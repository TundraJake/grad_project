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


def strip_list_newline(str_to_strip):
	for ii in range(len(str_to_strip)):
		str_to_strip[ii] = str_to_strip[ii].strip()

def authenticate(keys):
	auth = tweepy.OAuthHandler(key[0], key[1])
	auth.set_access_token(key[2], key[3])
	api = tweepy.API(auth)
	return api

streamer = TweetStreamListener()
key = list(open('keys.txt'))
strip_list_newline(key)
api = authenticate(key)


myStream = tweepy.Stream(auth = api.auth, listener=streamer)
# Idea: Use newspapers and keywords to fill lists and listen
# I'll have to figure out what I need to do to get special keywords, this might 
# include a very generic approach.
# Use below of async connection, otherwise main thread is used. 
# myStream.filter(track=['python'], async=True) 
special_words = ['DOW', 'S&P500', 'APPL']
myStream.filter(track=special_words)