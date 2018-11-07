'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy.random as npr
import unittest
import nltk
import psycopg2

nltk.download('punkt')
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()



CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
CURS = CONN.cursor()

SELECT_QUERY = "SELECT text FROM \"Tweets\" limit 10;" 

npr.seed(1)
# fix random seed for reproducibility

class Test(unittest.TestCase):

	def test_simple_tweet_model(self):
		CURS.execute(SELECT_QUERY)
		tweets = CURS.fetchall()
		for tweet in tweets:
			tweet_text = stemmer.stem(tweet[0])
			bow = word_tokenize(tweet_text)
			print(bow)



if __name__ == '__main__':
	unittest.main()