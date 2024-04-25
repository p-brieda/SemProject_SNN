import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import time
from util import getDefaultHyperparams, extractBatch
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareDataSet import PrepareDataSet
from torch.utils.data import DataLoader
from network import Net, RSNNet
# %%

#if __name__ == '__main__':

hyperparams = getDefaultHyperparams()
hyperparams['batch_size'] = 20
hyperparams['train_val_timeSteps'] = 1200

prepared_dataset = PrepareDataSet(hyperparams)

# loading the training dataset and creating a DataLoader
Train_dataset = DataProcessing(hyperparams, prepared_dataset, mode='training')
trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), hyperparams['batch_size'])
train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=0)
print('Training data loaded')

# %%

device = torch.device('cpu')
if torch.cuda.is_available():
    print('GPU available')
    device = torch.device('cuda')
model = Net(hyperparams)
model.to(device)


strategy1_start = time.time()

for epoch in range(1):
    for i, trial_iter in enumerate(train_loader):

        data, targets, errWeights = extractBatch(trial_iter, device)
        #print('Trial number: ', i, ' Data shape: ', data.shape, ' Targets shape: ', targets.shape, ' Error weights shape: ', errWeights.shape)
        output = model(data)


strategy1_end = time.time()
tot_batches = len(train_loader)*50

print('Strategy 1 time: ', strategy1_end - strategy1_start)
   


    
    
    
    
    
    
    
    
    
    
# INFINITE ITERATOR VERSION
'''
Finite_loader = create_Dataloaders(hyperparams, days=np.arange(10), mode='training')
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
'''

    


# %%
