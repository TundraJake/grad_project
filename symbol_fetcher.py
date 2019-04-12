'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''

import pandas as pd
from pandas_datareader import data as web
import numpy as np 
import datetime as dt
import sqlalchemy
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import numpy as np


import os

STOCK_SYMBOLS = ['AAPL']
FINANCE_SOURCE_NAME = 'yahoo'
data_frames = [] 

DIR = 'results/time_series_plots/'
START_DATE = '2016-04-02'
END_DATE = '2016-06-15'

PREPARED_DATASET_FILE_LOCATION = 'data/apple_data.xlsx'



def get_result_file_count():
    cpt = sum([len(files) for r, d, files in os.walk(DIR)])
    return cpt

def set_dataframes(start, end):
    for ii in range(len(STOCK_SYMBOLS)):
        data_frames.append(web.DataReader(STOCK_SYMBOLS[ii], FINANCE_SOURCE_NAME, start, end))
        data_frames[ii].astype(np.float32)

def fetch_stock_data():
    engine = sqlalchemy.create_engine('postgres://stock:money@localhost/stock_market_data')
    con = engine.connect()

    for ii in range(len(data_frames)):
        data_frames[ii] = data_frames[ii].reset_index()
        data_frames[ii]['Date'] = data_frames[ii]['Date'].dt.date
        data_frames[ii].to_sql(""+STOCK_SYMBOLS[ii]+"_ticker", con, if_exists='replace')

    prepared_tweets_ = pd.read_excel(PREPARED_DATASET_FILE_LOCATION, 'Stream')
    prepared_tweets_ = prepared_tweets_[['Tweet Id', 'Tweet content', 'Is a RT', 'Date']]
    prepared_tweets_.to_sql('apple_data', con, if_exists='replace')

def plot_data():

    for ii in range(len(data_frames)):
            
        lows = [item for item in data_frames[ii].loc[:, "Low"]]
        highs = [item for item in data_frames[ii].loc[:, "High"]]
        closes = [item for item in data_frames[ii].loc[:, "Close"]]
        volumes = [item for item in data_frames[ii].loc[:, "Volume"]]

        tmp_dates = data_frames[ii].reset_index().loc[:, "Date"]
        dates = []
        for jj in range(len(tmp_dates)):
            dates.append(tmp_dates.ix[jj])

        num_dates = [item for item in range(len(dates))]

        
        plt.title(STOCK_SYMBOLS[ii] + " Ticker")
        
        plt.ylabel('Price in $')
        plt.xlabel('Dates')

        plt.plot(dates, lows, label="Low")
        plt.plot(dates, closes, label="Close")
        plt.plot(dates, highs, label="High")
        plt.xticks(rotation=45)

        ax=plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))

        plt.legend()
        plt.gcf().subplots_adjust(bottom=0.28)

        plt.savefig(DIR + STOCK_SYMBOLS[ii] + '_' +START_DATE + '_' + END_DATE)

        plt.show()

def run():
    set_dataframes(START_DATE, END_DATE)
    fetch_stock_data()
    plot_data()

run()