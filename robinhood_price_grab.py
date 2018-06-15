import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style

#style.use('ggplot')

start_time = dt.datetime(2012,1,12)
end_time = dt.datetime.now()

data_frame = web.DataReader('QQQ', 'robinhood', start_time, end_time)
closing_prices = data_frame['close_price'].values

print(closing_prices)
data_frame.to_csv('price3.csv')  