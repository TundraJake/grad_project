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
import matplotlib as mpl
mpl.rcParams['font.size'] = 13.0
import matplotlib.pyplot  as plt

from settings import *

class TweetFrame(object):

    def __init__(self, symbol):
        # Init
        csv_file = PRICE_RAW_DIR + symbol + DOTCSV
        tweets_folder = TWEET_PREPROCESSED_DIR + symbol+ '/' 

        self.__symbol_ = symbol
        self.__time_series_ = self.__set_price_data(symbol, csv_file)
        self.__user_tweets_ = self.__set_tweet_data(symbol, tweets_folder)

        # Finalize a prepared dataset
        self.__processed_so_far_ = pd.DataFrame()

        pos_vals, neg_vals, word_counts = self.__extract_tweet_features()
        self.__user_tweets_['pos_val'] = pos_vals
        self.__user_tweets_['neg_val'] = neg_vals

        self.__processed_so_far_['pos_val'] = pos_vals
        self.__processed_so_far_['neg_val'] = neg_vals
        self.__processed_so_far_['word_count'] = word_counts
        self.__processed_so_far_['date'] = self.__user_tweets_['created_at'].dt.date

        self.__convert_to_daily_segments()
        self.__merge_time_series()

        self.__final_df_ = None
        self.__create_directories()
        self.__add_next_day_close_column()
        self.__add_movement_column()
        self.__delete_columns()

        self.__build_experiment_directory()
        self.__save_dataframe_to_file()
        self.__plot_daily_tweet_sentiment_graphs()

    def get_symbol(self):
        return self.__symbol_

    def get_post_processing_directory(self):
        return POST_PROCESSING_DIR + self.get_symbol() + '/' + self.get_symbol()

    def get_results_directory(self):
        return RESULTS_DIR + 'sentiments/' + self.get_symbol() + '/'

    def get_post_processing_directory(self):
        return POST_PROCESSING_DIR + self.get_symbol() + '/'

    def __create_results_directory_for_symbol(self):
        path = self.get_results_directory()
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                print(f'Cannot create directory at path: {path} ')
                print(f'Exiting Program...')
                sys.exit()

    def __create_post_processing_directory_for_symbol(self):
        path = self.get_post_processing_directory()
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                print(f'Cannot create directory at path: {path} ')
                print(f'Exiting Program...')
                sys.exit()

    def __create_directories(self):
        self.__create_results_directory_for_symbol()
        self.__create_post_processing_directory_for_symbol()
    
    def __plot_both_sentiment_graphs(self):
        df = self.__processed_so_far_
        pos_sentiments = df[df['daily_pos_sent_avg'] > POSITIVE_THRESHOLD]
        neg_sentiments = df[df['daily_pos_sent_avg'] <= POSITIVE_THRESHOLD]
        sizes = [len(pos_sentiments), len(neg_sentiments)]
        labels = ['Positive Sentiment', 'Negative Sentiment']
        colors = ['blue', 'orange']
        plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Sentiment Percentages for ' + self.get_symbol())

        plt.savefig(self.get_results_directory() + 'both_sent_graph')
        plt.clf()

    def __plot_positive_sentiment_averages(self):
        pos_sentiments = self.__processed_so_far_['daily_pos_sent_avg']
        plt.plot(pos_sentiments, color='blue')
        plt.title('Daily Averages of Positive Tweet Sentiment for ' + self.get_symbol())
        plt.xlabel('Trading Day')
        plt.ylabel('Sentiment')
        plt.axis([0, len(pos_sentiments), 0, 1])
        plt.savefig(self.get_results_directory() + 'pos_sent_graph')
        plt.clf()

    def __plot_negative_sentiment_averages(self):
        neg_sentiments = self.__processed_so_far_['daily_neg_sent_avg']
        plt.plot(neg_sentiments, color='orange')
        plt.title('Daily Averages of Negative Tweet Sentiment for ' + self.get_symbol())
        plt.xlabel('Trading Day')
        plt.ylabel('Sentiment')
        plt.axis([0, len(neg_sentiments), 0, 1])
        plt.savefig(self.get_results_directory() + 'neg_sent_graph')
        plt.clf()

    def __plot_daily_tweet_sentiment_graphs(self):
        self.__plot_positive_sentiment_averages()
        self.__plot_negative_sentiment_averages()
        self.__plot_both_sentiment_graphs()

    def __delete_columns(self):
        self.__processed_so_far_ = self.__processed_so_far_.drop(columns=['date'])


    def __add_movement_column(self):
        closes = self.__processed_so_far_['Close'].values.tolist()
        next_closes = self.__processed_so_far_['next_day_close'].values.tolist()

        movements = []

        assert (len(closes) == len(next_closes)), print(f'{len(closes)} != {len(next_closes)}')

        for ii, _ in enumerate(closes):
            if closes[ii] >= next_closes[ii]:
                movements.append(1)
            if closes[ii] < next_closes[ii]:
                movements.append(0)
        # Average the last two 

        self.__processed_so_far_['movements'] = np.array(movements)


    def __add_next_day_close_column(self):
        list_day_closes = self.__processed_so_far_['Close'].values.tolist()
        list_next_day_closes = list_day_closes[1:]
        # Average the last two 
        list_next_day_closes.append((list_day_closes[-1] + list_day_closes[-2])/2)
        self.__processed_so_far_['next_day_close'] = list_next_day_closes

    def __set_prediction_column(self, last_column):
        LAST_COLUMN_POSITION = -1
        cols = self.__processed_so_far_.columns.tolist()
        while cols[LAST_COLUMN_POSITION] != last_column:
            cols = cols[LAST_COLUMN_POSITION:] + cols[:LAST_COLUMN_POSITION]
        self.__processed_so_far_ = self.__processed_so_far_[cols]

    def __save_dataframe_to_file(self):
        outfile = self.get_post_processing_directory()
        for exp in EXPERIMENTS:
            self.__pull_columns(EXPERIMENTS[exp])
            self.__set_prediction_column(EXPERIMENTS[exp][-1])
            vals = np.array(self.__final_df_.values)
            np.save(outfile + exp + '_data', vals)
            np.savetxt(outfile + exp + '_data', vals, delimiter=',')

    def __pull_columns(self, columns_to_pull):
        vals = self.__processed_so_far_[columns_to_pull].values        
        self.__final_df_ = pd.DataFrame(vals)

    def __merge_time_series(self):
        self.__time_series_.rename(columns={'Date': 'date'}, inplace=True)

        ### Ensure the types are consistent. If this is not called, merge will not work.
        self.__processed_so_far_.date = self.__processed_so_far_.date.astype(str)
        self.__time_series_.date = self.__time_series_.date.astype(str)

        df = pd.merge(self.__processed_so_far_, self.__time_series_, how='inner')
        df = df.dropna()
        df = df.reset_index(drop=True)

        self.__processed_so_far_ = df
                
    def __convert_to_daily_segments(self):
        df = self.__processed_so_far_
        dates = sorted(df.date.unique())
        tmp_df = pd.DataFrame(columns=['daily_pos_sent_avg', 'daily_neg_sent_avg', 'date'])
        for index, date in enumerate(dates): 
            rows = df.loc[df['date'] == date]
            pos_total = rows['pos_val'].sum()
            neg_total = rows['neg_val'].sum()
            num_rows = rows.shape[0]
            pos_avg = pos_total/num_rows
            neg_avg = neg_total/num_rows
            # TODO: time this
            tmp_df = tmp_df.append({'daily_pos_sent_avg': pos_avg, 
                            'daily_neg_sent_avg': neg_avg, 
                            'date': date}, ignore_index=True)

        self.__processed_so_far_ = tmp_df
            
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
        df = self.__user_tweets_
        num_of_rows = df.shape[0]
        num_of_pos_tweets = len(df[df['pos_val'] > POSITIVE_THRESHOLD ].values.tolist())
        pos_ratio = num_of_pos_tweets / num_of_rows
        neg_ratio = P100 - pos_ratio 
        return pos_ratio, neg_ratio

    def print_tweets_statistics(self):
        pos_ratio, neg_ratio = self.__calculate_pos_neg_ratio()
        print('Number of tweets: ', len(self.__user_tweets_))
        print('Dataframe stats before final dataset: ', self.__processed_so_far_.describe())
        print(f'Percent Tweets that are positive given > {POSITIVE_THRESHOLD}: ', pos_ratio)
        print(f'Percent Tweets that are negative given <= {POSITIVE_THRESHOLD}: ', neg_ratio)
        columns = self.__processed_so_far_.columns
        print("Columns in dataset: ", columns)
        print("Final output shape: ", self.__final_df_.shape)
    
    def __build_experiment_directory(self):
        # Prepare for NNs. 
        for sym in SYMBOLS:
            for exp in EXPERIMENTS:
                path = NN_RESULT_DIR + exp + '/' + sym + '/'
                if not os.path.exists(path):
                    try:
                        os.makedirs(path)
                    except:
                        print(f'Cannot create directory at path: {path} ')
                        print(f'Exiting Program...')
                        sys.exit()   

    def extract_tweet_sentiments(self):
        POS_SENTIMENT = 1
        NEG_SENTIMENT = 2

        tb = Blobber(analyzer=NaiveBayesAnalyzer())
        pos_values = []
        neg_values = []
        num_words = []
 
        for tweet in self.__user_tweets_['text'].tolist():
            tweet = ' '.join(tweet)
            #Sentiment object.
            textblob = tb(tweet)
            sentiment = textblob.sentiment
            pos_values.append(sentiment[POS_SENTIMENT])
            neg_values.append(sentiment[NEG_SENTIMENT])
            num_words.append(len(textblob.words))
        
        return pos_values, neg_values, num_words

    def __extract_tweet_features(self):
        return self.extract_tweet_sentiments()
        
for sym in SYMBOLS:
    data = TweetFrame(sym)
    data.print_tweets_statistics()