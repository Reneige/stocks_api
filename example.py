# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:29:55 2023

@author: Rene Alby

# For this script to work, you must create a free Polygon.io API Key and add it 
to the Config.py file. Without this, the script will return query errors.

"""

import stocks_api
from stocks_api.stocks import market_data, analysis
import numpy as np


sp500 = market_data.refresh_sp500_list()
tl = market_data.get_tickerlist()

apple_price_series = market_data.get_stock_series('AAPL','2022-10-21','2023-10-01')
google_price_chart = market_data.get_stock_close_price_chart('GOOG','2022-05-21','2023-10-01')
apple_news_stream = market_data.get_news('aapl')


analyze = analysis()

price_matrix= analyze.price_matrix(analyze.random_ticker(),
                                   analyze.random_ticker(),
                                   analyze.random_ticker(),
                                   analyze.random_ticker(),
                                   analyze.random_ticker(),
                                   analyze.random_ticker(),
                                   start_date='2022-12-31', 
                                   end_date='2023-12-31'
                                  )

daily_returns = price_matrix.pct_change().dropna()
correlation_matrix = daily_returns.corr()
covariance_matrix = daily_returns.cov()
indexed_returns = (daily_returns + 1).cumprod() - 1
expected_return = (daily_returns + 1).product() - 1
st_dev = daily_returns.std()
equal_weights =  np.array( [1 / len(st_dev) for x in range(len(st_dev))] )

# Matrix form calculations for portfolio stdev / expected return taken from : https://en.wikipedia.org/wiki/Modern_portfolio_theory
portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(covariance_matrix, equal_weights)))
expected_portfolio_return = np.dot(expected_return.T, equal_weights)



