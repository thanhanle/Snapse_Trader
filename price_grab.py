#! env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pandas_datareader import data
import datetime

def pricegrab(stock,period):
    start_date = datetime.date(2012,1,12)
    end_date = datetime.date.today()
    data_source = 'yahoo' #download data from yahoo finance
    panel_data = data.DataReader(stock, data_source, start_date, end_date)
    prices = panel_data['Close'].values
    print('loading ' + stock)
    #dtprices = signal.detrend(prices)
    time.sleep(1)
    return panel_data

Y = pricegrab('MU','1Y') # Petroleum Brazil stock
Y.to_csv('price2.csv')  # save csv file 