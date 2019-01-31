'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''
# Create first network with Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.models import Sequential
from keras.layers import Dense
import numpy.random as npr
import unittest
import nltk
# import psycopg2

# nltk.download('punkt')
# from nltk.tokenize import word_tokenize

# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()



# CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
# CURS = CONN.cursor()

# SELECT_QUERY = "SELECT text FROM \"Tweets\" limit 10;" 

npr.seed(1)
# fix random seed for reproducibility

class Test(unittest.TestCase):

	def test_(self):

		# define the document
		text = 'The quick brown fox jumped over the lazy dog.'
		# estimate the size of the vocabulary
		words = set(text_to_word_sequence(text))
		vocab_size = len(words)
		print(vocab_size)
		# integer encode the document
		result = one_hot(text, round(vocab_size*1.3))
		print(result)



		# define 5 documents
		docs = ['Well done!',
				'Good work',
				'Great effort',
				'nice work',
				'Excellent Excellent Excellent Excellent Excellent Excellent!']
		
		# create the tokenizer
		t = Tokenizer()
		
		# fit the tokenizer on the documents
		t.fit_on_texts(docs)
		
		# summarize what was learned
		print(t.word_counts)
		print(t.document_count)
		print(t.word_index)
		print(t.word_docs)
		
		# integer encode documents
		encoded_docs = t.texts_to_matrix(docs, mode='count')
		print(encoded_docs)


if __name__ == '__main__':
	unittest.main()