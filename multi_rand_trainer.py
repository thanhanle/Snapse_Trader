
import numpy as np
import pandas as pd
import random



class sliding_trainer():
    def __init__(self, prices, stepsize, lookback):
        self.prices = prices
        self.stepsize = stepsize
        self.lookback = lookback

    def random_train(self):
        train = np.array([])
        position = random.randint(self.lookback,len(self.prices[0])-self.lookback+self.stepsize)
        for stock in self.prices:
            train = np.hstack([train,np.array(stock['Close'][position-self.lookback:position].values)])
            #print(train)

        test = self.prices[0]['Close'][position:position + self.stepsize].values
        #print(len(train))
        return train, test


#prices = [pd.read_csv('QQQ.csv'),pd.read_csv('DAX.csv')]
#st = sliding_trainer(prices,10,50)
#this, that = st.random_train()
#print(this,that)