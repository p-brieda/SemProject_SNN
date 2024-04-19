import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import random
from util import getDefaultArgs
from DataProcessing import DataProcessing
from torch.utils.data import DataLoader
from DayBatchSampler import DayBatchSampler




if __name__ == '__main__':

    args = getDefaultArgs()
    args['batchSize'] = 15

    # loading the training dataset and creating a DataLoader
    train_data = DataProcessing(args, mode='training')
    trainDayBatch_Sampler = DayBatchSampler(train_data.getDaysIdx(), args['batchSize'])
    train_loader = DataLoader(train_data, batch_sampler = trainDayBatch_Sampler , num_workers=0)

    '''
    # loading the validation dataset and creating a DataLoader
    val_data = DataProcessing(args, mode='validation')
    valDayBatch_Sampler = DayBatchSampler(val_data.getDaysIdx(), args['batchSize'])
    val_loader = DataLoader(val_data, batch_sampler = valDayBatch_Sampler , num_workers=0)

    # loading the test dataset and creating a DataLoader
    test_data = DataProcessing(args, mode='testing')
    testDayBatch_Sampler = DayBatchSampler(test_data.getDaysIdx(), args['batchSize'])
    test_loader = DataLoader(test_data, batch_sampler = testDayBatch_Sampler , num_workers=0)
    '''
    # OPTIONS WITH TRIALS FROM DIFFERENT DAYS INTO EACH BATCH
    #train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, num_workers=0)
    #val_loader = DataLoader(val_data, batch_size=args['batchSize'], shuffle=True, num_workers=0)
    #test_loader = DataLoader(test_data, batch_size=args['batchSize'], shuffle=True, num_workers=0)

    iterator = iter(train_loader)
    for k in range(7):
        trial_iter = next(iterator)
        print(trial_iter['neuralData'].shape)
        print(trial_iter['dayIdx'])
