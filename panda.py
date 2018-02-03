#!/usr/bin/env python3.6

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
	np.random.seed(1)

	# Creates a a series of 100 random values.
	s = pd.Series(np.random.randn(100))

	for item in s:
		print(item)

	print(s.values)
	print('', s.shape)
	print('value counts', s.value_counts)

	df = pd.DataFrame()

	for i in range(3):
		df+=pd.DataFrame(pd.Series(np.arange(i,i+10)))

	print(df.shape)

	start = dt.datetime(2010,1,1)
	end = dt.datetime(2012,12,30)

	item = web.DataReader('MSFT', 'yahoo', start, end)
	print(item)