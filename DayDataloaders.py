# SINGLE-DAY DATALOADER APPROACH

from PrepareData import PrepareData
from transforms import extractSentenceSnippet, addMeanNoise, addWhiteNoise
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import torch
import time
import logging
import  numpy as np
import math

# Class for preparing the single-day dataset for one of the three modes: training, validation or testing

class DayDataProcessing(Dataset):
    def __init__(self, hyperparam, prepared_data, mode='training'):
        self.hyperparams = hyperparam
        self.mode = mode
        self.prepared_data = prepared_data
        
        # selecting different amount of time steps for training/validation and testing
        if self.mode == 'training' or self.mode =='validation':
            self.timeSteps = self.hyperparams['train_val_timeSteps']
        else: # self.mode == 'testing'
            self.timeSteps = self.hyperparams['test_timeSteps']

        self.trials = self.prepared_data.getDatasets(mode=self.mode)




    def isViableDay(self):
        return len(self.trials['neuralData']) > 0




    def __len__(self):
        return len(self.trials['neuralData'])
    


        
    def __getitem__(self, idx):

        if self.mode == 'training':
            # extracting the trial from the training dataset
            trial = {key: data[idx] for key, data in self.trials.items()}
            
            # extracting random snippets from the sentences
            extractSentenceSnippet(trial, self.timeSteps, self.hyperparams['directionality'])
            if self.hyperparams['constantOffsetSD'] > 0 or self.hyperparams['randomWalkSD'] > 0:
                # adding mean noise to the trial
                addMeanNoise(trial, self.hyperparams['constantOffsetSD'], self.hyperparams['randomWalkSD'],self.timeSteps)
            if self.hyperparams['whiteNoiseSD'] > 0:
                # adding white noise to the trial
                addWhiteNoise(trial, self.hyperparams['whiteNoiseSD'], self.timeSteps)

        elif self.mode == 'validation':
            trial = {key: data[idx] for key, data in self.trials.items()}
            # extracting random snippets from the sentences without adding noise
            extractSentenceSnippet(trial, self.timeSteps, self.hyperparams['directionality'])

        else: # self.mode == 'testing'
            # no snippets are extracted, the whole trial is used and no noise is added
            trial = {key: data[idx] for key, data in self.trials.items()}

        return trial
    



class DaySampler(Sampler):
    def __init__(self, len_dataset, batch_size, shuffle, fill_batch=True):
        self.Indeces = np.arange(len_dataset)
        self.batch_size = batch_size
        self.fill_batch = fill_batch
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            shuffled_indeces = np.random.permutation(self.Indeces)
        else: 
            shuffled_indeces = self.Indeces
        for i in range(0, len(shuffled_indeces), self.batch_size):
            batch = shuffled_indeces[i:i+self.batch_size]
            if self.fill_batch and len(batch) < self.batch_size:
                #batch = np.concatenate((batch, shuffled_indeces[:self.batch_size-len(batch)]))
                batch = np.concatenate([batch, np.random.choice(shuffled_indeces, self.batch_size-len(batch), replace=True)])
            yield batch

    def __len__(self):
        return math.ceil(len(self.Indeces)/self.batch_size)




# Class for creating the dataloaders for the training, validation and testing datasets for all the days
class create_Dataloaders:
    def __init__(self, manual, hyperparam, days, mode):
        self.datasets = []
        self.samplers = []
        prepared_data_dir = hyperparam['prepared_data_dir']

        if mode == 'training':
            shuffle = True
            fill_batch = True
        else: # mode == 'validation' or mode == 'testing'
            shuffle = False
            fill_batch = False

        if not os.path.exists(prepared_data_dir + 'prepared_data_days.pth') or (manual and input('Do you want to recompute the prepared data? (y/n) ') == 'y'):
            print('Preparing data')
            self.prepared_datasets = []
            for day in days:
                prepared_data = PrepareData(hyperparam, days=[day])
                self.prepared_datasets.append(prepared_data)
                self.datasets.append(DayDataProcessing(hyperparam, prepared_data, mode))
                self.samplers.append(DaySampler(len(self.datasets[day]), hyperparam['batch_size'], shuffle=shuffle, fill_batch=fill_batch))
                
            torch.save(self.prepared_datasets, prepared_data_dir + 'prepared_data_days.pth')
            print('Data saved')
            
        else:
            print(f"Loading prepared data from dir")
            logging.info(f"Loading prepared data from dir")
            self.prepared_datasets = torch.load(prepared_data_dir + 'prepared_data_days.pth')
            for day in days:
                self.datasets.append(DayDataProcessing(hyperparam, self.prepared_datasets[day], mode))
                self.samplers.append(DaySampler(len(self.datasets[day]), hyperparam['batch_size'], shuffle=shuffle, fill_batch=fill_batch))
            print(f"Data loaded")

        self.dataloaders = []
        self.viabledays = []
        

        for day in days:
            if self.datasets[day].isViableDay():
                self.viabledays.append(day)
                self.dataloaders.append(DataLoader(self.datasets[day], num_workers=0, batch_sampler=self.samplers[day]))
                

    
    def getDataloaders(self):
        return self.dataloaders
    
    def getViableDays(self):
        return self.viabledays
    



# Class for creating the infinite iterators for the dataloaders (one for each day)
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
        
        
