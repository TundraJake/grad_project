'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

'''

import tweepy
from tweepy import API
import twitter
import psycopg2
from dateutil.parser import parse
import datetime

tweets = []

CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
CURS = CONN.cursor()
INSERT_QUERY = """INSERT INTO \"Tweets\" 
				(text, location, user_id, retweeted, retweet_count, 
				favorite_count, reply_count, quote_count, utc_date_created) 
				VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

INSERT_USER_QUERY = """INSERT INTO \"User_Tweets\" 
				(text, location, user_id, retweeted, retweet_count, 
				favorite_count, utc_date_created) 
				VALUES (%s, %s, %s, %s, %s, %s, %s)"""

class TweetStreamListener(tweepy.StreamListener):

	def __init__(self, api=None):
		super(tweepy.StreamListener).__init__()
		self.BATCH_COUNTER = 0
		self.api = api or API()
	
	def on_status(self, status):
		print(status._json)
		print('\n'*20)
		retweeted_full_text = status._json.get('retweeted_status')
		retweeted = False

		try:
			text = status.extended_tweet['full_text']
			# print(text)
			retweeted = True
		except AttributeError:
			text = status.text

		date = status.created_at
		# date = datetime.strftime('%Y-%M-%D')


		tweets.append(
					(text, 
					status.user.location, 
					status.id, 
					retweeted,
					status.retweet_count,
					status.favorite_count,
					status.reply_count,
					status.quote_count,
					str(date)))

		print(len(tweets))
		if len(tweets) == 10:
			print('Inserting Batch: ' + str(self.BATCH_COUNTER))
			self.BATCH_COUNTER += 1
			self.write_to_postgres(tweets)
		

	def on_error(self, status_code):
		if status_code == 420:
			#returning False in on_data disconnects the stream
			return False
	
	def write_to_postgres(self, data):
		for tweet in data:
			try:
				CURS.execute(INSERT_QUERY, tweet)
			except psycopg2.Error as error:
				print(error)
		CONN.commit()
		tweets.clear()


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

def stream():
	streamer = TweetStreamListener()
	api = authenticate()

	myStream = tweepy.Stream(auth=api.auth, listener=streamer)

	special_words = ['AAPL', 'Apple', 'market', 'stocks', 'bull', 'bullish', 'bear', 'bearish']

	myStream.filter(languages=["en"], track=special_words)





def on_status(statuses):
	tweets = []
	for status in statuses: 
		full_text = status._json.get('full_text')
		print(full_text)
		retweeted = False

		try:
			retweeted = status._json.get('retweeted')
			print(retweeted)
		finally:
			print('yuppers')

		date = status.created_at
		# date = datetime.strftime('%Y-%M-%D')

		tweets.append(
					(full_text, 
					status.user.location, 
					status.id, 
					retweeted,
					status.retweet_count,
					status.favorite_count,
					str(date)))

	write_to_postgres(tweets)

def write_to_postgres(tweets):
	for tweet in tweets:
		try:
			CURS.execute(INSERT_USER_QUERY, tweet)
		except psycopg2.Error as error:
			# Likely over 300 chars.
			print(error)

	CONN.commit()
	tweets.clear()

def get_users_tweets():
	api = authenticate()

	# Both former and present
	apple_executives = [1636590253, 17104751, 754500, 20832113, 38854434,
						22938914, ]

	apple_pundits_analysts = [12131132, 33423, 9324442, 1835411, 14331688, 15439395, 134665872]

	# 44764761 special case The Loop Media
	apple_journalists = [2960721, 44764761]

	users = apple_executives + apple_pundits_analysts + apple_journalists
	users_tweets = []  

	# Maximum 200 tweets may be extracted at a time, starting with earliest. 
	number_of_tweets=200
	for user in users:
		users_tweets.append(api.user_timeline(user_id=user, count=number_of_tweets, tweet_mode='extended')) 

	# create array of tweet information: username,  
	# tweet id, date/time, text
	for tweets in users_tweets: 
		# print(tweets)
		on_status(tweets)


def main():
	# stream()
	get_users_tweets()

if __name__ == '__main__':
	main()









