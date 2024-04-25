# %%

import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import time
from util import getDefaultArgs
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareDataSet import PrepareDataSet
from torch.utils.data import DataLoader



if __name__ == '__main__':

    args = getDefaultArgs()
    args['batchSize'] = 20

    prepared_dataset = PrepareDataSet(args)
    
    # loading the training dataset and creating a DataLoader
    Train_dataset = DataProcessing(args, prepared_dataset, mode='training')
    trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), args['batchSize'])
    train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=0)

    

    
    k=1
    strategy1_start = time.time()
    
    for epoch in range(1):
        for i, trial_iter in enumerate(train_loader):
            #print(k)
            k += 1
            print(trial_iter['neuralData'].shape)
            print(trial_iter['dayIdx'])

    strategy1_end = time.time()
    tot_batches = len(train_loader)*50
    
    print('Strategy 1 time: ', strategy1_end - strategy1_start)
    print('Total batches: ', tot_batches)




    
    # INFINITE ITERATOR VERSION
    Finite_loader = create_Dataloaders(args, days=np.arange(10), mode='training')
    train_loaders = Finite_loader.getDataloaders()
    viable_train_days = Finite_loader.getViableDays()
    train_InfIterators = DayInfiniteIterators(train_loaders)

    strategy2_start = time.time()
    
    tot_batches = 30
    for k in range(tot_batches):
        dayIdx = np.random.choice(viable_train_days)
        next_iter = train_InfIterators.getNextIter(dayIdx)
        #print(k)
        print(next_iter['neuralData'].shape)
        print(next_iter['dayIdx'])

    strategy2_end = time.time()
    print('Strategy 2 time: ', strategy2_end - strategy2_start)

    

