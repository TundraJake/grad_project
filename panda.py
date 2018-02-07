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
import pymysql

# padnas uses numpy, set seed for entire pandas library. 

def run():
    df = pd.DataFrame()

    start = dt.datetime(2010,1,1)
    end = dt.datetime(2012,12,30)

    item = web.DataReader('MSFT', 'morningstar', start, end)
    print(item)