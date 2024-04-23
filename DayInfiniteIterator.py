import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from DayDataProcessing import DayDataProcessing

class DayInfiniteIterator:

    def __init__(self, args, days, mode):
        self.args = args
        self.mode = mode
        self.batch_size = args['batchSize']
        self.datasets = []

        self.dataloadersTrain = []
        self.dataloadersVal = []
        self.dataloadersTest = []
        self.dataloadersAll = [self.dataloadersTrain, self.dataloadersVal, self.dataloadersTest]

        self.iteratorsTrain = []
        self.iteratorsVal = []
        self.iteratorsTest = []
        self.iteratorsAll = [self.iteratorsTrain, self.iteratorsVal, self.iteratorsTest]

        for dayIdx in days:
            self.datasets.append(DayDataProcessing(self.args, dayIdx, mode=self.mode))
            [anyTrainTrial, anyValTrial] = self.datasets[-1].isViableDay()

            if self.mode == 'training' or self.mode == 'validation':
                Shuffle = True
                if anyTrainTrial:
                    self.datasets[-1].setMode('training')
                    self.dataloadersTrain.append(DataLoader(self.datasets[-1], batch_size=self.batch_size, shuffle=Shuffle, num_workers=0))
                    self.iteratorsTrain.append(iter(self.dataloadersTrain[-1]))

                if anyValTrial:
                    self.datasets[-1].setMode('validation')
                    self.dataloadersVal.append(DataLoader(self.datasets[-1], batch_size=self.batch_size, shuffle=Shuffle, num_workers=0))
                    self.iteratorsVal.append(iter(self.dataloadersVal[-1]))

            else:
                Shuffle = False

                self.datasets[-1].setMode('testing')
                self.dataloadersTest.append(DataLoader(self.datasets[-1], batch_size=self.batch_size, shuffle=Shuffle, num_workers=0))
                self.iteratorsTest.append(iter(self.dataloadersTest[-1]))


    def getNextIter(self, dayIdx, mode):
        self.datasets[dayIdx].setMode(mode)
        modeIdx = ['training', 'validation', 'testing'].index(mode)
        try:
            return next(self.iteratorsAll[modeIdx][dayIdx])
        except StopIteration:
            self.iteratorsAll[modeIdx][dayIdx] = iter(self.dataloadersAll[modeIdx][dayIdx])
            return next(self.iteratorsAll[modeIdx][dayIdx])
        

    def getViableDays(self):
        viableTrainDays = []
        viableValDays = []

        for dayIdx in range(len(self.datasets)):
           [isTrainViable, isValViable] = self.datasets[dayIdx].isViableDay()
           if isTrainViable: viableTrainDays.append(dayIdx)
           if isValViable: viableValDays.append(dayIdx)

        return [viableTrainDays, viableValDays]




