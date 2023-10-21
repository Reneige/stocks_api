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
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt


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


# implement some portfolio optimization based off the random tickers
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
rf=0.5

# define a function to optimize (here the sharpe ratio)
def sharpe_ratio(weights):
    vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    er = np.dot(expected_return.T, weights)
    return (er - rf)/vol
    
        
# set the boundry for weights between zero and 1 (i.e. no short selling)
bnds = Bounds(lb=0, ub=1)

# set the objective function to sum(x) - 1 to convert a minimization into a maximization
constraints_def = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})

max_sharpe = minimize(sharpe_ratio,
                      equal_weights,
                      bounds=bnds,
                      constraints=constraints_def,
                      method='SLSQP'
                      )

# retrieve optimal weights according to 
optimal_weights = max_sharpe['x']
check = 1 - sum(optimal_weights) == 0


#produce 1000 random weight vectors
list_of_vectors=[]
for _ in range(1000):
    x = np.random.dirichlet(np.ones(5),size=1)
    x = np.concatenate(x)
    list_of_vectors.append(x)
    
# produce 1000 random returns and volatilities to plot
p_return = []
p_vol =[]

for weightvector in list_of_vectors:
    portfolio_volatility = np.sqrt(np.dot(weightvector.T, np.dot(covariance_matrix, weightvector)))
    expected_portfolio_return = np.dot(expected_return.T, weightvector)

    p_vol.append(portfolio_volatility)
    p_return.append(expected_portfolio_return)
    
title = 'Portfolio of: ' + ', '.join(random_tickerlist)    
# plot various random potential portfolio returns versus volatilities    
plt.scatter(p_vol, p_return)
plt.title(title)
plt.xlabel("volatility")
plt.ylabel("expected return")
plt.show()

# plot the indexed returns of the underlying assets
indexed_returns.plot()
