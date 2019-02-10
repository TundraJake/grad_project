'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''

import pandas as pd
# from pandas_datareader import data as web
import numpy as np 
import datetime as dt
import sqlalchemy
import quandl

class Ticker_Fetcher():

	def __init__(self, tickers, data_engine, start_date, end_date):
		self._tickers = tickers	
		self._data_engine = data_engine
		self._data_frames = []
		self._sql_engine = sqlalchemy.create_engine('postgres://stock:money@localhost/stock_market_data')
		self._start_date = start_date
		self._end_date = end_date

	def __fetch_stock_data(self):
		self._data_frames = quandl.get("WIKI/AAPL", start_date=self._start_date, end_date=self._end_date)

	def get_data_frames(self):
		self.__fetch_stock_data()
		return self._data_frames

