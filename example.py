# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:29:55 2023

@author: Rene Alby

# For this script to work, you must create a free Polygon.io API Key and add it 
to the Config.py file. Without this, the script will return query errors.

"""

#import stocks_api
from stocks_api.stocks import market_data, analysis


# get the list of S&P 500 companies, create a tickerlist
sp500 = market_data.refresh_sp500_list()
tl = market_data.get_tickerlist()


# get some random data from the API
apple_price_series = market_data.get_stock_series('AAPL','2022-10-21','2023-10-01')
google_price_chart = market_data.get_stock_close_price_chart('GOOG','2022-05-21','2023-10-01')
apple_news_stream = market_data.get_news('aapl')


# create random list of tickers
analyze = analysis()
random_tickerlist = analyze.n_random_tickers(5)

# generate a price matrix of 1 year stock prices from n random tickers 
price_matrix= analyze.price_matrix(random_tickerlist,
                                   start_date='2022-09-30', 
                                   end_date='2023-09-30'
                                  )

