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
import os

STOCK_SYMBOLS = ['MSFT', 'AAPL', 'NVDA']
NUMERICAL_FINANCE_SOURCE_NAME = 'yahoo'
data_frames = [] 

DIR = 'results/time_series_plots/'
START_DATE = '2017-03-17'
END_DATE = '2018-03-21'

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
        data_frames[ii].reset_index().plot(kind='line', 
            title=STOCK_SYMBOLS[ii], 
            x='Date', 
            y=['Close', 'High', 'Low'], 
            color=['red', 'blue', 'green']).get_figure().savefig(DIR + STOCK_SYMBOLS[ii] + '_' +START_DATE + '_' + END_DATE)

        plt.show()


def run():
    set_dataframes(START_DATE, END_DATE)
    fetch_stock_data()
    plot_data()

run()