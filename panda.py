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

# pandas uses numpy, set seed for entire pandas library. 

STOCK_SYMBOLS = ['MSFT', 'AAPL', 'NVDA']
NUMERICAL_FINANCE_SOURCE_NAME = 'yahoo'
data_frames = [] 
engine = sqlalchemy.create_engine('postgres://stock:money@localhost/stock_market_data')

def fetch_stock_data(start, end):
    con = engine.connect()
    for ii in range(len(STOCK_SYMBOLS)):
        data_frames.append(web.DataReader(STOCK_SYMBOLS[ii], 'yahoo', start, end))
        data_frames[ii].astype(np.float32)
        data_frames[ii].to_sql(""+STOCK_SYMBOLS[ii]+"_ticker", con, if_exists='replace')

