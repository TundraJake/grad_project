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
NUMERICAL_FINANCE_SOURCE_NAME = 'yahoo'
data_frames = [] 

DIR = 'results/time_series_plots/'
START_DATE = '2019-01-01'
END_DATE = '2019-03-30'

def get_result_file_count():
    cpt = sum([len(files) for r, d, files in os.walk(DIR)])
    return cpt

def set_dataframes(start, end):
    for ii in range(len(STOCK_SYMBOLS)):
        data_frames.append(web.DataReader(STOCK_SYMBOLS[ii], 'yahoo', start, end))
        data_frames[ii].astype(np.float32)

def fetch_stock_data():
    engine = sqlalchemy.create_engine('postgres://stock:money@localhost/stock_market_data')
    con = engine.connect()
    for ii in range(len(data_frames)):
        data_frames[ii].to_sql(""+STOCK_SYMBOLS[ii]+"_ticker", con, if_exists='replace')

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
        
        print(dates)
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