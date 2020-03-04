import pandas as pd
pd.set_option('display.expand_frame_repr', False)
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

DATA_DIR = './data/stocknet-dataset/'
SAVE_DIR = './data/post_processing/'
PRICE_PREPROCESSED_DIR = DATA_DIR + 'price/preprocessed/'
PRICE_RAW_DIR = DATA_DIR + 'price/raw/'
TWEET_PREPROCESSED_DIR = DATA_DIR + 'tweet/preprocessed/'
TWEET_RAW_DIR = DATA_DIR + 'tweet/raw/'

class TweetFrame(object):

    def __init__(self, symbol):
        # Init
        csv_file = PRICE_RAW_DIR + symbol + DOTCSV
        tweets_folder = TWEET_PREPROCESSED_DIR + symbol+ '/' 

        self.symbol_ = symbol
        self.times_series_ = self.__set_price_data(symbol, csv_file)
        self.user_tweets_ = self.__set_tweet_data(symbol, tweets_folder)

        # Finalize a prepared dataset
        self.processed_so_far_ = pd.DataFrame()

        pos_vals, neg_vals = self.__determine_tweet_sentiment()
        self.user_tweets_['pos_val'] = pos_vals
        self.user_tweets_['neg_val'] = neg_vals

        self.processed_so_far_['pos_val'] = pos_vals
        self.processed_so_far_['neg_val'] = neg_vals
        self.processed_so_far_['date'] = self.user_tweets_['created_at'].dt.date

        self.__convert_to_daily_segments()
        self.__merge_time_series()

        self.final_df_ = None
        self.__normalize_dataframe()
        self.__write_processed_symbol_pickle_file()

    def __write_processed_symbol_pickle_file(self):
        outfile = SAVE_DIR + self.symbol_
        vals = np.array(self.final_df_.values)
        np.save(outfile, vals)

    def __normalize_dataframe(self):
        vals = self.processed_so_far_.values
        min_max_scalar = MinMaxScaler()
        scaled_vals = min_max_scalar.fit_transform(vals)
        self.final_df_ = pd.DataFrame(scaled_vals)

    def __merge_time_series(self):
        self.times_series_.rename(columns={'Date': 'date'}, inplace=True)

        ### Ensure the types are consistent. If this is not called, merge will not work.
        self.processed_so_far_.date = self.processed_so_far_.date.astype(str)
        self.times_series_.date = self.times_series_.date.astype(str)

        df = pd.merge(self.processed_so_far_, self.times_series_, how='inner')
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df.drop(columns=['date'])

        self.processed_so_far_ = df
                
    def __convert_to_daily_segments(self):
        df = self.processed_so_far_
        dates = sorted(df.date.unique())
        tmp_df = pd.DataFrame(columns=['daily_pos_avg', 'daily_neg_avg', 'date'])
        for index, date in enumerate(dates): 
            rows = df.loc[df['date'] == date]
            pos_total = rows['pos_val'].sum()
            neg_total = rows['neg_val'].sum()
            num_rows = rows.shape[0]
            pos_avg = pos_total/num_rows
            neg_avg = neg_total/num_rows
            # TODO: time this
            tmp_df = tmp_df.append({'daily_pos_avg': pos_avg, 
                            'daily_neg_avg': neg_avg, 
                            'date': date}, ignore_index=True)

        self.processed_so_far_ = tmp_df
            
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
            return pd.read_csv(file, sep=",")
        except:
            print(f'Error in __set_price_data for {symbol}.')
            print(sys.exc_info())
            sys.exit()
    
    def __calculate_pos_neg_ratio(self):
        P100 = 1.0
        df = self.user_tweets_
        num_of_rows = df.shape[0]
        num_of_pos_tweets = len(df[df['pos_val'] > .5 ].values.tolist())
        pos_ratio = num_of_pos_tweets / num_of_rows
        neg_ratio = P100 - pos_ratio 
        return pos_ratio, neg_ratio

    def print_tweets_statistics(self):
        pos_ratio, neg_ratio = self.__calculate_pos_neg_ratio()
        print('Number of tweets: ', len(self.user_tweets_))
        print('Dataframe stats before final dataset: ', self.processed_so_far_.describe())
        print('Percent Tweets that are positive: ', pos_ratio)
        print('Percent Tweets that are negative: ', neg_ratio)
        print('Set so far: \n', self.processed_so_far_)
    
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