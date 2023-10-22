# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 06:59:15 2023

@author: Rene Alby

"""
    
import json
import requests
import pandas as pd
import random
from stocks_api.config import API_KEY
from stocks_api.delay import delay



DELAY_QUERY = delay()

class market_data:
    
    BASE_URL='https://api.polygon.io/'

    KEY_REF = '&apiKey='+API_KEY

    VERSION={2:'v2/', 
             3:'v3/'}

    TYPE={'AGG' : 'aggs/ticker/', 
          'TIC' : 'reference/tickers?',
          'OPT' : 'reference/options/contracts?',
          'NEW' : 'reference/news?'}    
    
    
    
    def __init__(self):
        pass

        
    def validate_ticker(ticker) -> str:
        ''' ensures ticker is string and no longer than 4 characters and upper case'''
        
        if not isinstance (ticker, str):
            ticker = str(ticker)
        if len(ticker) > 4:
            ticker = ticker[0:4]
        ticker = ticker.upper()
        return ticker
            
    
    def refresh_sp500_list() -> pd.DataFrame:
        ''' scrapes a list of S&p500 companies from wikipedia and returns as DataFrame '''
        
        page = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        dat = page.text
        try:
            pd_data = pd.read_html(dat)
        except:
            pd_data = pd.DataFrame()
    
        if isinstance(pd_data, list):    
            try:
                df = pd_data[0]
            except:
                df = pd.DataFrame()
        else:
            df = pd_data
        return df
    
    
    def custom_query(*args, **kwargs) -> str:
        ''' takes a list of args and kwargs and appens / to the args and & to the 
            kwards so you can construg a custom query. 
            
            custom_query('v3','reference', x='tickers',y='search=apple')
            returns : 'https://api.polygon.io/v3/reference/tickers&search=apple&apiKey=***'
            
        '''
        
        list_of_strings = [str(x)+"/" for x in args] + [str(x)+"&" for x in kwargs.values()]
        string = ''.join(x for x in list_of_strings)
        return (market_data.BASE_URL + string + market_data.KEY_REF)
    
    def request_to_dict(querystring: str) -> dict:
        ''' Queries API and returns data as Dict of values '''
        DELAY_QUERY.step()
        response = requests.get(querystring)
        remote_data = response.text
        data = json.loads(remote_data)
        
        # if query status is error, return blank dict
        if data['status'] == 'ERROR':
            print("Query Error")
            print(data)
            return {}
        return data    
    
    def request_to_df(querystring : str) -> pd.DataFrame:
        ''' Queries API and returns data as DataFrame'''
        
        j_data = market_data.request_to_dict(querystring)
        
        # if no data or error, return empty df
        if len(j_data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(j_data['results'])
        return df
    
    
    
    def validate(version    : int, 
                 query_type : str) -> bool :
        
        ''' to validate query input '''
        
        return ((version in market_data.VERSION.keys()) and (query_type in market_data.TYPE.keys()))  
 
    
 
    def query_builder(version       : int,
                      query_type    : str,
                      variable_data : str
                      )-> str:
        
        ''' builds a query to send to API for data '''
        
        if not market_data.validate(version, query_type):
            print('invalid input')
            return -1
        
        query = ( market_data.BASE_URL 
                + market_data.VERSION[version] 
                + market_data.TYPE[query_type] 
                + variable_data 
                + market_data.KEY_REF)
        
        return query
    
    
        
    def get_stock_series(ticker     : str, 
                         start_date : str, 
                         end_date   : str, 
                         dayfreq=1
                         ) -> pd.DataFrame:
        ''' constructs a query for the API to extract raw stock data. 
            
            Eg:
            
            ticker      : 'AAPL'
            start_date  : '2022-12-31'
            end_date    : '2023-12-31'
            dayfreq     :  1
            
            returns DataFrame
        '''
        
        ticker = market_data.validate_ticker(ticker)
        freq = str(dayfreq)
        d_range = start_date + '/' + end_date
        
        query_variable = (ticker 
                          + f'/range/{freq}/day/' 
                          + d_range 
                          + '?adjusted=true&sort=asc')
        
        query = market_data.query_builder(2, 'AGG', query_variable)
        data = market_data.request_to_df(query)
        
        if data.empty:
            return data
        
        data['date'] = pd.to_datetime(data['t'], unit='ms')
        data = data.set_index('date')
        data = data.drop(columns='t')
        return data
    
    
    
    def search_for_tickers(search_string):
        ''' searches ticker database for a string 
            eg: search_for_tickers(alpha) 
        
            returns DataFrame
        '''
        
        
        variable_data = f'search={search_string}&active=true'
        query = market_data.query_builder(3, 'TIC', variable_data)
        return market_data.request_to_df(query)
    
    def get_stock_close_price_chart(ticker     : str, 
                                    start_date : str, 
                                    end_date   : str,
                                    ) -> pd.DataFrame:
        
        ''' returns the stock price chart as a plot.
        
            eg:
            
            ticker      : 'AAPL'
            start_date  : '2022-12-31'
            end_date    : '2023-12-31'
            
            returns DataFrame
        '''
        ticker = market_data.validate_ticker(ticker)
        df = market_data.get_stock_series(ticker, start_date, end_date)
        
        if df.empty:
            return
    
        plt = df['c'].plot(color=['dimgrey'])
        plt.legend([ticker])
        plt
        return plt
    
    
    def get_options_contracts(ticker : str):
        
        ticker = market_data.validate_ticker(ticker)
        variable_data = f'underlying_ticker={ticker}'
        query = market_data.query_builder(3,'OPT', variable_data)
        return market_data.request_to_df(query)
 
    
    def get_news(ticker : str):
        ''' returns a dataframe of news items for a ticker'''
        
        ticker = market_data.validate_ticker(ticker)
        variable_data = f'ticker={ticker}'
        query = market_data.query_builder(2,'NEW', variable_data)
        return  market_data.request_to_df(query)
 
    
    def get_tickerlist():
        ''' returns a list of tickers from the S&P 500'''
        
        sp500 = market_data.refresh_sp500_list()
        return sp500['Symbol'].tolist()
 

class analysis:
    
    def __init__(self):
        self.sp500 = market_data.refresh_sp500_list()
        self.tickerlist = market_data.get_tickerlist()

    def random_ticker(self) -> str: 
        ''' returns a random ticker from the S&P 500'''
        
        ticker = random.choice(self.tickerlist)
        return ticker

    def n_random_tickers(self, n : int) -> list:
        ''' returns a list of n random tickers from the S&P 500.        
            Max value for n = 500
        '''
        
        # set ceiling of n at 100
        if  n>= 500:
            n = 500
        
        # using a set ensures all items are unique
        x = set()
        while len(x) < n:
            ticker = random.choice(self.tickerlist)
            x.add(ticker)
        return list(x)
    
    def price_matrix(self, ticker_list, start_date='2021-12-31', end_date='2022-12-31') -> pd.DataFrame:
        ''' returns a dataframe containing the price series of all the tickers fed in as a list'''
        
        # converting to set automatically removes duplicates
        tickers = set(ticker_list)
    
        items = []
                
        
        for ticker in tickers:
            df = market_data.get_stock_series(ticker, start_date, end_date)
            df = df[['c']]
            df = df.rename(columns={'c':ticker})
            items.append(df)
    
        data = items.pop()
        for _df in items:
            data = data.merge(_df, left_index = True, right_index=True)

        return data

