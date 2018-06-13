
import numpy as np
import pandas as pd
import random

prices = pd.read_csv('price.csv')

class sliding_trainer():
    def __init__(self, prices, stepsize, lookback):
        self.prices = prices
        self.stepsize = stepsize
        self.lookback = lookback

    def random_train(self):
        position = random.randint(self.lookback,len(self.prices)-self.lookback+self.stepsize)
        train = self.prices[position-self.lookback:position]
        test = self.prices[position:position + self.stepsize]
        return train.values, test.values




    