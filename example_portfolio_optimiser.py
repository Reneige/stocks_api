# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:17:55 2023

@author: Rene Alby

building off the stocks API, this script implements mean-variance optimization
(aka classical portfolio theory or markowitz portfolio theory) techniques to calculate the 
optimal portfolio by maximising the sharpe ratio.

In the sample below, 5 random stocks are selected from the S&P 500
the tool then outputs the optimal portfolio weights, as well as an efficient frontier
and a performance chart.

# For this script to work, you must create a free Polygon.io API Key and add it 
to the Config.py file. Without this, the script will return query errors.

"""

#import stocks_api
from stocks_api.stocks import analysis
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from datetime import date, timedelta

class portfolio_opmitiser:
    ''' instantiate optimiser by passing list of tickers as instance variable '''
    
    def __init__(self, ticker_list, start_date, end_date, risk_free_rate=0.05):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date        
        self.risk_free_rate = risk_free_rate                
        self.price_matrix = self.get_price_matrix()
        self.trading_days = len(self.price_matrix)
        self.daily_returns = self.get_daily_returns()
        self.covariance_matrix = self.get_covariance_matrix()
        self.correlation_matrix = self.get_correlation_matrix()
        self.expected_returns = self.get_annual_stock_returns_vector()

    # generate a price matrix of 1 year stock prices from n random tickers 
    def get_price_matrix(self):
        return analyze.price_matrix(self.ticker_list,
                                    start_date='2022-09-30', 
                                    end_date='2023-09-30')

    def get_daily_returns(self) -> pd.DataFrame:
        return self.price_matrix.pct_change().dropna()


    def get_covariance_matrix(self) -> pd.DataFrame:
        return self.daily_returns.cov()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        return self.daily_returns.corr()
    
    def get_stocks_volatility(self) -> pd.DataFrame:
        return self.daily_returns.std()
    
    def get_equal_weights_vector(self) -> np.array:
        ''' returns numpy vector (n,) of equal weights for each ticker in portfolio'''
        return np.array( [1 / len(self.ticker_list) for x in range(len(self.ticker_list))] )
    
    def get_annual_stock_returns_vector(self) -> pd.Series:
        ''' returns pandas Series (n,) of equal weights for each ticker in portfolio'''
        return (self.daily_returns + 1).product() - 1

    def get_annual_portfolio_volatility(self, weights_vector : np.array) -> float:
        ''' returns the portfolio volatility applying matrix calculation: vol = sqrt(wTpw) 
            where :
                w is weights vector
                wT is transposed weights vector
                p is the annualsed covariance matrix 
                
            note: to annualise the covariance matrix we multiply it by the number of observed
            trading days in our one-year returns series
        '''
        return np.sqrt(np.dot(weights_vector.T, np.dot(self.covariance_matrix * self.trading_days, weights_vector)))

    def get_portfolio_expected_return(self, weights_vector : np.array) -> float:
        ''' returns the portfolio expected return applying matrix calculation: erp = RTw 
            where :
                w is weights vector
                RT is transposed expected returns vector
                
                here the expected returns are 1-year trailing annual stock returns
        '''
        return np.dot(self.expected_returns.T, weights_vector)

    def portfolio_sharpe_ratio(self, weights_vector : np.array) -> float:
        ''' calculates the portfolio sharpe ratio given the weights held in each
            stock. These are passed as a numpy vector of weights in decimal form
        '''
        exp_return = self.get_portfolio_expected_return(weights_vector)
        vol = self.get_annual_portfolio_volatility(weights_vector)
        return (exp_return - self.risk_free_rate) / vol

    def invert_sharpe(self, weights_vector : np.array) -> float:
        ''' Background - SciPy optimizer contains a mimize optimization method. In order
            to maximize, we must send the inverted sharpe ratio to the optimizer 
        '''
        return -1 * self.portfolio_sharpe_ratio(weights_vector)

    def get_optimal_weights(self):

        # set the boundry for weights between zero and 1 (i.e. no short selling)
        bnds = Bounds(lb=0, ub=1)

        # For a minimzation we need the objective function to = 0 (so sum of weights -1).         
        constraints_def = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})

        max_sharpe = minimize(self.invert_sharpe,
                              self.get_equal_weights_vector(),
                              bounds=bnds,
                              constraints=constraints_def,
                              method='SLSQP'
                              )
        
        return max_sharpe['x']
    
    def plot_indexed_return_chart(self):
        ''' returns a matplotlib chart of the indexed daily returns '''
        
        indexed_returns = (self.daily_returns + 1).cumprod() - 1 
        return indexed_returns.plot()

    
    def plot_efficient_frontier(self):
        ''' plot an efficient frontier via brute force calculation 
            returns matplotlib.pyploy.scatter chart
        '''
        
        # get optimal weights
        opt_weights = self.get_optimal_weights()
        
        # produce 1000 random portfolios to plot
        
        # start with 1000 random weight vectors
        list_of_vectors=[]
        for _ in range(1000):
            x = np.random.dirichlet(np.ones(len(self.ticker_list)),size=1)
            x = np.concatenate(x)
            list_of_vectors.append(x)
        
        # then produce 1000 random returns and volatilities to plot
        p_return, p_vol = [], []

        for weightvector in list_of_vectors:
            portfolio_volatility = self.get_annual_portfolio_volatility(weightvector)
            expected_portfolio_return = self.get_portfolio_expected_return(weightvector)

            p_vol.append(portfolio_volatility)
            p_return.append(expected_portfolio_return)

        # get the optimal portfolio volatility and return
        opt_vol = self.get_annual_portfolio_volatility(opt_weights)
        opt_return = self.get_portfolio_expected_return(opt_weights)
        
        # create chart title
        title = 'Portfolio of: ' + ', '.join(self.ticker_list)    
        
        # plot random portfolios and then plot optimal portfolio in red  
        plt.scatter(p_vol, p_return)
        plt.scatter(opt_vol, opt_return, c='red', label='optimal portfolio')
        plt.legend()
        plt.xlabel("volatility")
        plt.ylabel("expected return")
        plt.title(title)
        return plt


# create strings of start and end dates looking over 1 year period starting yesterday
start_date = (date.today() - timedelta(days=366)).strftime("%Y-%m-%d")
end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# create random list of tickers
analyze = analysis()
random_tickerlist = analyze.n_random_tickers(4)

# isntantiate the optimizer, get optimal weights and produce charts
optimise = portfolio_opmitiser(random_tickerlist, start_date, end_date)
optimise.get_optimal_weights()
optimise.plot_efficient_frontier()
optimise.plot_indexed_return_chart()

