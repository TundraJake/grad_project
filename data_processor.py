from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, \
                                StandardScaler, \
                                LabelEncoder as le

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer

import re

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

DIR = 'data/preprocessed/'
PREPARED_DATASET_FILE = DIR + 'prepped_data_time_series'
PREPARED_DAILY_DATASET_FILE = DIR + 'prepped_data_daily_time_series'
FILE_TYPE = '.csv'

ENGINE = create_engine('postgresql://stock:money@localhost:5432/stock_market_data')
USER_QUERY = ''' SELECT * FROM \"User_Tweets\"; '''
NUMERICAL_QUERY = ''' SELECT  "Close", "Date", "Open", "High", "Low", "Volume" FROM \"AAPL_ticker\" WHERE EXTRACT(\'ISODOW\' FROM "Date") < 6 LIMIT 48; '''
DATA_QUERY = ''' SELECT * FROM prepared_data_3;'''
STREAMED_TWEETS_QUERY = ''' SELECT * FROM \"Tweets\"; '''

tb = Blobber(analyzer=NaiveBayesAnalyzer())

class Data_Processor(object):

    def __init__(self):
        # Dataframe objects
        self.user_tweets_ = pd.read_sql(DATA_QUERY, ENGINE)
        self.numerical_ = pd.read_sql(NUMERICAL_QUERY, ENGINE)
        self.streamed_tweets_ = pd.read_sql(STREAMED_TWEETS_QUERY, ENGINE)

        self.user_tweets_word_counts_ = None
        self.user_tweets_document_count_ = 0
        self.user_tweets_word_index_ = None
        self.user_tweets_word_docs_ = None
        self.vectorized_user_tweets_ = None

        self.user_tweets_sentiment_ = []

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

    def to_integer(self, dt_time):
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day

    def __clear_data(self, tweets, tweet_sentiment, vectorized_tweets):
        tweets = None
        tweet_sentiment = None
        vectorized_tweets = None

    def __build_file_format_string(self, filename, batch_num, file_set):
        return filename + file_set + str(batch_num) + FILE_TYPE

    def __build_user_dataframe_obect(self, filename):
 
        tweets = self.user_tweets_
        tweet_sentiment = self.user_tweets_sentiment_
        vectorized_tweets = self.vectorized_user_tweets_,

        training_batches = 8
        testing_batches = 2

        number_of_tweets = len(tweet_sentiment)

        train_size = int(number_of_tweets  * .8)
        test_size = number_of_tweets - train_size

        training_batch_list = [int(train_size / training_batches) * item for item in range(1, training_batches + 1)]
        testing_batch_list = [int(test_size / testing_batches) * item + train_size for item in range(1, testing_batches + 1)]

        # print(int(len(tweet_sentiment)), training_batch_list)
        # print(int(len(tweet_sentiment)), testing_batch_list)

        scalar = MinMaxScaler(feature_range=(0,1))


        pos_values, neg_values, clafficiations = self.__get_sentiment_values(tweet_sentiment)
        tmp = []        
        train_batch = 0
        test_batch = 0
        
        for index, _ in enumerate(pos_values):
            print("Progress: " + str(round(index / number_of_tweets * 100, 1)) )
            
            cur_date = tweets.loc[index, 'Date']
            print(cur_date)
            '''li = np.array([
                    self.to_integer(tweets.loc[index, 'Date']),
                    tweets.loc[index, 'Open'],  
                    tweets.loc[index, 'Low'],
                    tweets.loc[index, 'High'],
                    tweets.loc[index, 'Volume'],
                    pos_values[index], 
                    neg_values[index] ]  
                    # clafficiations[index] ] 
                    + vectorized_tweets[index].tolist()
            )
            print(li)
            tmp.append(li)
            
            if index in training_batch_list:
                # tmp = scalar.fit(tmp)
                print("\rSaving training batch " + str(train_batch) + "...")
                temp_array = np.array(tmp)
                np.savetxt(self.__build_file_format_string(filename, train_batch, '_csv_batch'), temp_array, delimiter=",")
                np.save(self.__build_file_format_string(filename, train_batch, '_training_'), temp_array)
                train_batch += 1
                tmp = []

            elif index in testing_batch_list:
                # tmp = scalar.fit(tmp)
                print("\rSaving testing batch " + str(test_batch) + "...")
                temp_array = np.array(tmp)
                
                test_batch += 1
                tmp = []'''
                
            
        #self.__clear_data(tweets, tweet_sentiment, vectorized_tweets)


    def __build_daily_tweet_sentiment(self, tweets, tweet_sentiment, vectorized_tweets, filename):
        scalar = MinMaxScaler(feature_range=(0,1))

        self.user_tweets_ = self.user_tweets_.set_index(['Date'])
        self.user_tweets_ = self.user_tweets_.drop(columns=['Close', 'Volume', 'High', 'Open', 'Low'])

        pos_values, neg_values, clafficiations = self.__get_sentiment_values(tweet_sentiment)    
        length = len(self.numerical_)
        temp_array = []  

        if (len(self.user_tweets_.groupby('Date').count().values) != len(self.numerical_)):
            print('Counts for dates does not equal counts for time series dates...')
            exit

        print(self.user_tweets_.groupby('Date').count().values)
        total = 0
        pos_sent_date_ranges = self.user_tweets_.groupby('Date').count().values.tolist()

        for item in pos_sent_date_ranges:
            total += item[0]

        print(total)

        pos_so_far = 0
        position = 0
        daily_sentiments = []
        for pos_range in pos_sent_date_ranges:
            for value in range(pos_range[0]):
                pos_so_far += pos_values[position]
                position += 1
            
            daily_sentiments.append(pos_so_far/pos_range[0])
            pos_so_far = 0

        print(daily_sentiments)


        for index in range(length):


            print("Progress: " + str(round(index / length * 100, 1)) )

            date = self.numerical_.loc[index, 'Date']
            print(date)

            li = np.array([
                    self.numerical_.loc[index, 'Close'],
                    self.to_integer(date),
                    self.numerical_.loc[index, 'Open'],
                    self.numerical_.loc[index, 'Low'],
                    self.numerical_.loc[index, 'High'],
                    self.numerical_.loc[index, 'Volume'],
                    daily_sentiments[index],
                    pos_sent_date_ranges[index][0]

            ])
            temp_array.append(li)

        # np.savetxt(self.__build_file_format_string(filename, 0, ''), temp_array, delimiter=",")
        # np.save(self.__build_file_format_string(filename, 0, '_full_set_2'), temp_array)            


    def __build_dataframe_objects(self):
        self.__build_user_dataframe_obect(PREPARED_DATASET_FILE)


    def process(self):
        self.__clean_tweets()
        self.__build_dataframe_objects()
        #self.__build_daily_tweet_sentiment(self.user_tweets_, self.user_tweets_sentiment_, self.vectorized_user_tweets_, PREPARED_DAILY_DATASET_FILE)


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


data = Data_Processor()
data.process()