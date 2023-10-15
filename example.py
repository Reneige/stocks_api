# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:29:55 2023

@author: Renea
"""

import stocks_api
from stocks_api.stocks import market_data, analysis



sp500 = market_data.refresh_sp500_list()
tl = market_data.get_tickerlist()

test1 = market_data.get_stock_series('AAPL','2022-10-21','2023-10-01')
test2 = market_data.get_stock_close_price_chart('GOOG','2022-05-21','2023-10-01')
test3 = market_data.get_news('aapl')


analyze = analysis()



test7 = analyze.price_matrix(analyze.random_ticker(),
                             analyze.random_ticker(),
                             analyze.random_ticker(),
                             analyze.random_ticker(),
                             analyze.random_ticker(),
                             analyze.random_ticker()
                                  
                                  )

