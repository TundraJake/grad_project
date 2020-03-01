import pandas as pd
from sklearn.preprocessing import MinMaxScaler, \
                                StandardScaler, \
                                LabelEncoder as le

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer

import re
import os

from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import numpy as np

import sys
import datetime

DOTCSV = '.csv'
DOTTXT = '.txt'

DATA_DIR = 'data/stocknet-dataset/'
PRICE_PREPROCESSED_DIR = DATA_DIR + 'price/preprocessed/'
TWEET_PREPROCESSED_DIR = DATA_DIR + 'tweet/preprocessed/'
TWEET_RAW_DIR = DATA_DIR + 'tweet/raw/'

tb = Blobber(analyzer=NaiveBayesAnalyzer())

class TweetFrame(object):

    def __init__(self, symbol):
        # Init
        csv_file = PRICE_PREPROCESSED_DIR + symbol + DOTTXT
        tweets_folder = TWEET_PREPROCESSED_DIR + symbol+ '/' 

        self.times_series_ = self.__set_price_data(symbol, csv_file)
        self.user_tweets_ = self.__set_tweet_data(symbol, tweets_folder)

        # Finalize a prepared dataset
        self.nn_ready_data = pd.DataFrame()

        pos_vals, neg_vals = self.__determine_tweet_sentiment()
        self.nn_ready_data['pos_val'] = pos_vals
        self.nn_ready_data['neg_val'] = neg_vals
        self.nn_ready_data['date'] = self.user_tweets_['created_at'].dt.date
        
    def __format_date(self):
        return self.user_tweets_['created_at'].date()
            
    def __set_tweet_data(self, symbol, folder):
        try:
            # Store in list first.
            tmp_list = []
            for filename in os.listdir(folder):
                df = pd.read_json(folder + filename, lines = True)
                tmp_list.append(df)
            return pd.concat(tmp_list, axis=0, ignore_index = True)

        except FileNotFoundError:
            print(f"""
            No file for symbol \"{symbol}\". 
            File name/location: {folder}""")
            sys.exit()
       
        except:
            print(f"Error in __set_tweet_data for {symbol}.")
            print(sys.exc_info()[1])
            sys.exit()

    def __set_price_data(self, symbol, file):
        try:
            return pd.read_csv(file, sep=" ")
        except:
            print(f'Error in __set_price_data for {symbol}.')
            print(sys.exc_info())
            sys.exit()
    
    def __calculate_pos_neg_ratio(self):
        P100 = 1.0
        df = self.nn_ready_data
        num_of_rows = df.shape[0]
        num_of_pos_tweets = len(df[df['pos_val'] > .5 ].values.tolist())
        pos_ratio = num_of_pos_tweets / num_of_rows
        neg_ratio = P100 - pos_ratio 
        return pos_ratio, neg_ratio

    def print_tweets_statistics(self):
        pos_ratio, neg_ratio = self.__calculate_pos_neg_ratio()
        print('Number of tweets: ', len(self.user_tweets_))
        print('Document Count:', self.user_tweets_document_count_)
        print('Dataframe stats: ', self.nn_ready_data.describe())
        print('Percent Tweets that are positive: ', pos_ratio)
        print('Percent Tweets that are negative: ', neg_ratio)
    
    def __determine_tweet_sentiment(self):
        POS_SENTIMENT = 1
        NEG_SENTIMENT = 2

        tb = Blobber(analyzer=NaiveBayesAnalyzer())
        pos_values = []
        neg_values = []
 
        for tweet in self.user_tweets_['text'].tolist():
            tweet = ' '.join(tweet)
            #Sentiment object.
            sentiment = tb(tweet).sentiment
            pos_values.append(sentiment[POS_SENTIMENT])
            neg_values.append(sentiment[NEG_SENTIMENT])

        return pos_values, neg_values


data = TweetFrame('AAPL')
data.print_tweets_statistics()