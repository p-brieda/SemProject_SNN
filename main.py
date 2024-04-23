# %%

import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import time
from util import getDefaultArgs
from DataProcessing import DataProcessing
from torch.utils.data import DataLoader
from CustomBatchSampler import CustomBatchSampler
from DayInfiniteIterator import DayInfiniteIterator
#from DayDataProcessing import DayDataProcessing
from torch.utils.data import BatchSampler, SequentialSampler


# %%

if __name__ == '__main__':

    args = getDefaultArgs()
    args['batchSize'] = 30
    
    
    # loading the training dataset and creating a DataLoader
    TrainVal_dataset = DataProcessing(args, mode='training')
    trainDayBatch_Sampler = CustomBatchSampler(TrainVal_dataset.getDaysIdx(), args['batchSize'])
    train_loader = DataLoader(TrainVal_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=0)

    
    # loading the validation dataset and creating a DataLoader
    #dataset.setMode('validation')
    #valDayBatch_Sampler = CustomBatchSampler(dataset.getDaysIdx(), args['batchSize'])
    #val_loader = DataLoader(dataset, batch_sampler = valDayBatch_Sampler , num_workers=0)
    
    k=1
    strategy1_start = time.time()
    
    for epoch in range(50):
        for i, trial_iter in enumerate(train_loader):
            #print(k)
            k += 1
            #print(trial_iter['neuralData'].shape)
            #print(trial_iter['dayIdx'])

    strategy1_end = time.time()
    tot_batches = len(train_loader)*50
    
    print('Strategy 1 time: ', strategy1_end - strategy1_start)
    print('Total batches: ', tot_batches)






    # INFINITE ITERATOR VERSION
    day_InfIterator = DayInfiniteIterator(args, days=np.arange(10), mode='training')

    [trainViableDays, valViableDays] = day_InfIterator.getViableDays()
    strategy2_start = time.time()
    
    for k in range(tot_batches):
        dayIdx = np.random.choice(trainViableDays)
        next_iter = day_InfIterator.getNextIter(dayIdx, 'training')
        #print(k)
        #print(next_iter['neuralData'].shape)
        #print(next_iter['dayIdx'])

    strategy2_end = time.time()
    print('Strategy 2 time: ', strategy2_end - strategy2_start)

