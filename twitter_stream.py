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
from tweepy import API
import panda
import twitter
import psycopg2

tweets = []
CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
CURS = CONN.cursor()
INSERT_QUERY = "INSERT INTO \"Tweets\" (text, location, user_id, retweeted) VALUES (%s, %s, %s, %s)"

class TweetStreamListener(tweepy.StreamListener):

	def __init__(self, api=None):
		super(tweepy.StreamListener).__init__()
		self.BATCH_COUNTER = 0
		self.api = api or API()
	
	def on_status(self, status):
		retweeted_full_text = status._json.get('retweeted_status')
		retweeted = False
		try:
			text = status.extended_tweet["full_text"]
			retweeted = True
		except AttributeError:
			text = status.text

		tweets.append((text, status.user.location, status.id, retweeted))

		if len(tweets) == 100:
			self.write_to_postgres(tweets)
		

	def on_error(self, status_code):
		if status_code == 420:
			#returning False in on_data disconnects the stream
			return False
	
	def write_to_postgres(self, data):
		print("Inserting Batch: " + str(self.BATCH_COUNTER))
		for tweet in data:
			try:
				CURS.execute(INSERT_QUERY, tweet)
			except psycopg2.Error as error:
				print(error)
				print(tweet)
		CONN.commit()
		tweets.clear()
		self.BATCH_COUNTER += 1

# Remove newlines. 
def strip_list_newline(str_to_strip):
	for ii in range(len(str_to_strip)):
		str_to_strip[ii] = str_to_strip[ii].strip()

# Authenticates to Jacob's market Twitter app.
def authenticate():
	key = list(open('keys.txt'))
	strip_list_newline(key)
	auth = tweepy.OAuthHandler(key[0], key[1])
	auth.set_access_token(key[2], key[3])
	api = tweepy.API(auth)
	return api

def main():
	streamer = TweetStreamListener()
	api = authenticate()

	myStream = tweepy.Stream(auth=api.auth, listener=streamer)
	# Idea: Use newspapers and keywords to fill lists and listen
	# I'll have to figure out what I need to do to get special keywords, this might 
	# include a very generic approach.
	# Use below of async connection, otherwise main thread is used. 
	# myStream.filter(track=['python'], async=True) 
	special_words = ['NVDA', 'AAPL', 'Apple', 'market', 'stocks', 'prices']

	# Used http://boundingbox.klokantech.com/ to filter NYC tweets about the stock market. 
	# May use multiple locations, filter by news sources/people/etc. 
	NYC = [-122.48, 47.79, -79.15, -122.48]
	# Location, because it's not well documented in the source, takes two coordinate position per point (4 total).
	# Still does not filter everything not related to the stock market. 


	# TODO: Filter by english documents.   
	myStream.filter(languages=["en"], track=special_words)
	# myStream.userstream(track=special_words)

if __name__ == '__main__':
	main()









