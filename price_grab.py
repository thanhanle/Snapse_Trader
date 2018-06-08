#! env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pandas_datareader import data
import datetime

def pricegrab(stock,period):
    start_date = datetime.date(2017,1,12)
    end_date = datetime.date.today()
    data_source = 'yahoo'
    panel_data = data.DataReader(stock, data_source, start_date, end_date)
    prices = panel_data['Close'].values
    print('loading ' + stock)
    #dtprices = signal.detrend(prices)
    time.sleep(1)
    return prices

Y = pricegrab('MSFT','1Y')
print(Y)