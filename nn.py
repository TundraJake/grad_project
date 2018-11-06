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

CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
CURS = CONN.cursor()

SELECT_QUERY = "SELECT * FROM \"Tweets\";" 

npr.seed(1)
# fix random seed for reproducibility

class Test(unittest.TestCase):

	def test_simple_numerical_test(self):
		print("Creating Sequential Model...")
		CURS.execute(SELECT_QUERY)
		yuppers = CURS.fetchall()
		print(yuppers)
		model = Sequential()


if __name__ == '__main__':
	unittest.main()