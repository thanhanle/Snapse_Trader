
import numpy as np
import pandas as pd

prices = pd.read_csv('price.csv')

class sliding_trainer():
    def __init__(self, prices, stepsize, lookback):
        self.prices = prices
        self.stepsize = stepsize
        self.lookback = lookback
        self.position = self.lookback
        self.inputs = self.prices[self.position - self.lookback:self.position]
        self.testdata = self.prices[self.position-1:self.position + self.stepsize]

    def slidestep(self):
        if (self.position + self.stepsize) <= len(self.prices):
            self.position += self.stepsize
        else:
            self.position = self.lookback
            return "done"
        self.inputs = self.prices[self.position-self.lookback:self.position]
        self.testdata = self.prices[self.position-1:self.position + self.stepsize]
        return self.inputs.values, self.testdata.values




    