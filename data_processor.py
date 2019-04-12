from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://stock:money@localhost:5432/stock_market_data')
USER_QUERY = ''' SELECT * FROM \"User_Tweets\"; '''
NUMERICAL_QUERY = ''' SELECT "Date", "Close", "High", "Low", "Volume" FROM \"AAPL_ticker\"; '''
DATA_QUERY = ''' SELECT * FROM prepared_data_3;'''
STREAMED_TWEETS_QUERY = ''' SELECT * FROM \"Tweets\"; '''

from sklearn.preprocessing import MinMaxScaler, \
                                StandardScaler, \
                                LabelEncoder as le

scaler = MinMaxScaler(feature_range = (0, 1))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer

import re

from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import numpy as np

import sys
import datetime

tb = Blobber(analyzer=NaiveBayesAnalyzer())

CLASSIFICATION = 0
POS_SENTIMENT = 1
NEG_SENTIMENT = 2

# create CountVectorizer object
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

DIR = 'data/csv/'
USER_FILE = DIR +'user/user_data'
STREAMED_FILE = DIR + 'streamed/streamed_data'
PREPARED_DATASET_FILE_LOCATION = 'data/apple_data.xlsx'
PREPARED_DATASET_FILE = DIR + 'prepared/prepped_data'
FILE_TYPE = '.npy'

class Data_Processor(object):

    def __init__(self):
        # Dataframe objects
        self.user_tweets_ = pd.read_sql(DATA_QUERY, engine)
        self.numerical_ = pd.read_sql(NUMERICAL_QUERY, engine)
        self.streamed_tweets_ = pd.read_sql(STREAMED_TWEETS_QUERY, engine)
        self.prepared_tweets_ = pd.read_sql(DATA_QUERY, engine)

        self.user_tweets_word_counts_ = None
        self.user_tweets_document_count_ = 0
        self.user_tweets_word_index_ = None
        self.user_tweets_word_docs_ = None
        self.vectorized_user_tweets_ = None


        self.user_tweets_sentiment_ = []

    #
    #
    # Helper functions
    #
    #

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
        print(self.numerical_)
    
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

    #
    #
    # User Tweets
    #
    #

    def __get_tweet_sentiment(self, tweets):
        tb = Blobber(analyzer=NaiveBayesAnalyzer())
        # List of Sentiment objects
        return [tb(tweet).sentiment for tweet in tweets]
    
    def __get_sentiment_values(self, sentiment_list):
        POS = 1
        NEG = -1
        
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

    def to_integer(self, dt_time):
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day

    def __clear_data(self, tweets, tweet_sentiment, vectorized_tweets):
        tweets = None
        tweet_sentiment = None
        vectorized_tweets = None

    def __build_file_format_string(self, filename, batch_num, file_set):
        return filename + file_set + str(batch_num) + FILE_TYPE

    def __build_user_dataframe_obect(self,tweets, tweet_sentiment, vectorized_tweets, filename):
 
        training_batches = 10
        testing_batches = 5

        train_size = int(len(tweet_sentiment)  * .8)
        test_size = len(tweet_sentiment) - train_size

        training_batch_list = [int(train_size / training_batches) * item for item in range(1, training_batches + 1)]
        testing_batch_list = [int(test_size / testing_batches) * item + train_size for item in range(1, testing_batches + 1)]

        print(int(len(tweet_sentiment)), training_batch_list)
        print(int(len(tweet_sentiment)), testing_batch_list)
        print(tweets)
        import time 
        time.sleep(3)

        pos_values, neg_values, clafficiations = self.__get_sentiment_values(tweet_sentiment)
        tmp = []        
        length = len(pos_values)
        train_batch = 0
        test_batch = 0

        print(tweets)
        print(len(tweets))

        for index, _ in enumerate(pos_values):
            # print(type(tweets.loc[index, 'Date']))
            # print(index)
            # print(tweets.loc[index, 'Date'])
            print("Progress: " + str(round(index / length * 100, 2)) )

            li = np.array([pos_values[index], 
                    neg_values[index], 
                    clafficiations[index], 
                    self.to_integer(tweets.loc[index, 'Date']) ] + vectorized_tweets[index].tolist())
            tmp.append(li)

            if index in training_batch_list:
                print("Training...")
                temp_array = np.array(tmp)
                np.savetxt(self.__build_file_format_string(filename, train_batch, '_training_'), temp_array, delimiter=",")
                train_batch += 1
                tmp = []

            elif index in testing_batch_list:
                print("Testing...")
                temp_array = np.array(tmp)
                np.savetxt(self.__build_file_format_string(filename, test_batch, '_testing_'), temp_array, delimiter=",")
                test_batch += 1
                tmp = []
            
        self.__clear_data(tweets, tweet_sentiment, vectorized_tweets)


    def __build_dataframe_objects(self):
        self.__build_user_dataframe_obect(self.user_tweets_, self.user_tweets_sentiment_, self.vectorized_user_tweets_, PREPARED_DATASET_FILE)




    def process(self):
        self.__analyze_user_tweets()
        # self.__analyze_streamed_tweets()
        
        self.__build_dataframe_objects()

    def __analyze_user_tweets(self):
        tweets = self.user_tweets_['text']
        tweets = tweets.dropna()
        print(len(tweets))
        tweets = self.__remove_tweet_urls(tweets)
        self.user_tweets_['text'] = tweets
        self.vectorized_user_tweets_ = self.__vectorize_tweets(tweets)
        self.user_tweets_sentiment_ = self.__get_tweet_sentiment(tweets)
        self.__summarize_user_tweets(tweets)


data = Data_Processor()
data.process()