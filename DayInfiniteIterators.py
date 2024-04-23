import torch
import numpy as np
import scipy.io

class DayInfiniteIterators:

    def __init__(self, dataloaders):

        self.dataloaders = dataloaders
        self.iterators = []

        for day in range(len(dataloaders)):
            self.iterators.append(iter(self.dataloaders[day]))



    def getNextIter(self, dayIdx):
        try:
            return next(self.iterators[dayIdx])
        except StopIteration:
            self.iterators[dayIdx] = iter(self.dataloaders[dayIdx])
            return next(self.iterators[dayIdx])
        




