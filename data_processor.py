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

vectorizer = CountVectorizer(stop_words=["i", "me", "my", "myself", "we", "our", "ours", 
"ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
"she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", 
"themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", 
"are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", 
"doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", 
"by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", 
"above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
"further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
"few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
"own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

DOTCSV = '.csv'
DOTTXT = '.txt'

DATA_DIR = 'data/stocknet-dataset/'
PRICE_PREPROCESSED_DIR = DATA_DIR + 'price/preprocessed/'
TWEET_PREPROCESSED_DIR = DATA_DIR + 'tweet/preprocessed/'

tb = Blobber(analyzer=NaiveBayesAnalyzer())

class Data_Processor(object):

    def __init__(self, symbol):
        csv_file = PRICE_PREPROCESSED_DIR + symbol + DOTTXT
        tweets_folder = TWEET_PREPROCESSED_DIR + symbol+ '/' 

        self.symbol_times_series = None
        try:
            self.symbol_times_series = pd.read_csv(csv_file, sep=" ")
        except:
            print(f"Error for {symbol}.")
            print(sys.exc_info())
            sys.exit()

        self.user_tweets_ = None
        try:
            # Store in list first.
            tmp_list = []
            for filename in os.listdir(tweets_folder):
                df = pd.read_json(tweets_folder + filename, lines = True)
                tmp_list.append(df)

            self.user_tweets_ = pd.concat(tmp_list, axis=0, ignore_index = True)

        except FileNotFoundError:
            print(f"""
            No file for symbol \"{symbol}\". 
            File name/location: {tweets_folder}""")
            sys.exit()
       
        except:
            print(f"Error for {symbol}.")
            print(sys.exc_info())
            sys.exit()
        
        self.user_tweets_word_counts_ = None
        self.user_tweets_document_count_ = 0
        self.user_tweets_word_index_ = None
        self.user_tweets_word_docs_ = None
        self.vectorized_user_tweets_ = None

        self.user_tweets_sentiment_ = []

        # Sanity checks.
        assert self.user_tweets_.columns[0] == "created_at", f"Expected \"created_at\", got {self.user_tweets_.columns[0]}"


    def __summarize_user_tweets(self, tweets):
        t = Tokenizer()
        t.fit_on_texts(tweets)

        self.user_tweets_word_counts_ = t.word_counts
        self.user_tweets_document_count_ = t.document_count
        self.user_tweets_word_index_ = t.word_index
        self.user_tweets_word_docs_ = t.word_docs
    
    def print_tweet_counts(self):
        print("Select User Tweets", self.user_tweets_document_count_)

    def print_user_tweets(self):
        print(self.user_tweets_)

    def print_numerical(self):
        print(self.symbol_times_series)
    
    def print_streamed_tweets(self):
        print(self.streamed_tweets_)

    def print_vectorized_contents(self):
        print("User vector length: ", len(self.vectorized_user_tweets_))
        print("Streamed vector length: ", len(self.vectorized_streamed_tweets_))

    def __remove_tweet_urls(self, tweets):
        return [re.sub(r"- http\S+", "", tweet) for tweet in tweets]

    def __vectorize_tweets(self, tweets):
        vec = vectorizer.fit_transform(tweets)
        tmp_list = vec.toarray()
        return tmp_list

    def __get_tweet_sentiment(self, tweets):
        tb = Blobber(analyzer=NaiveBayesAnalyzer())
        # List of Sentiment objects
        return [tb(tweet).sentiment for tweet in tweets]
    
    def __get_sentiment_values(self, sentiment_list):
        POS = 1
        NEG = 0
        CLASSIFICATION = 0
        POS_SENTIMENT = 1
        NEG_SENTIMENT = 2

        pos_values = []
        neg_values = []
        clafficiations = []

        for sentiment in sentiment_list:
            if sentiment[CLASSIFICATION] == 'pos':
                clafficiations.append(POS)
                pos_values.append(sentiment[POS_SENTIMENT])
                neg_values.append(sentiment[NEG_SENTIMENT])
            else:
                clafficiations.append(NEG)
                neg_values.append(sentiment[NEG_SENTIMENT])
                pos_values.append(sentiment[POS_SENTIMENT])

        return pos_values, neg_values, clafficiations

    def __index_at_zero(self, dt_time):
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day

    def __build_file_format_string(self, filename, batch_num, file_set):
        return filename + file_set + str(batch_num) + DOTCSV

    def process(self):
        print()
        #self.__clean_tweets()

    def __clean_tweets(self):
        # Remove row and bad data if any.
        self.user_tweets_ = self.user_tweets_.drop(columns=['row'])
        self.user_tweets_ = self.user_tweets_.dropna()

        # Remove HTTPS from tweets.
        tweets = self.user_tweets_['text']
        tweets = self.__remove_tweet_urls(tweets)

        # Assign new text back to dataframe.
        self.user_tweets_['text'] = tweets
        self.vectorized_user_tweets_ = self.__vectorize_tweets(tweets)

        # List of tweet sentiments.
        self.user_tweets_sentiment_ = self.__get_tweet_sentiment(tweets)
        self.__summarize_user_tweets(tweets)


data = Data_Processor("AAPL")
data.process()