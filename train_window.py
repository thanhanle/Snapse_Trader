
import numpy as np
import pandas as pd

prices = pd.read_csv('price.csv')

class sliding_trainer():
    def __init__(self, prices, stepsize, lookback):
        self.prices = prices
        self.stepsize = stepsize
        self.lookback = lookback
        self.position = self.lookback
        #self.inputs = self.prices[self.position - self.lookback:self.position]
        #self.testdata = self.prices[self.position-1:self.position + self.stepsize]

    def slidestep(self):
        train = np.array([])
        if (self.position + self.stepsize) <= len(self.prices[0])-1:
            self.position += self.stepsize
        else:
            self.position = self.lookback
            return "done","done"

        for stock in self.prices:
            train = np.hstack([train,np.array(stock['Close'][self.position-self.lookback:self.position].values)])
        test = self.prices[0]['Close'][self.position:self.position + self.stepsize].values

        return train, test

