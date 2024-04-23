import os
from datetime import datetime
import random
import numpy as np
from util import prepareDataCubesForRNN, unfoldDataCube
from transforms import extractSentenceSnippet, addMeanNoise, addWhiteNoise
import torch
from torch.utils.data import Dataset




class DataProcessing(Dataset):
    def __init__(self, args, prepared_dataset, mode='training'):
        self.args = args
        self.mode = mode
        self.prepared_dataset = prepared_dataset

        if self.mode == 'training' or self.mode == 'validation':
            self.timeSteps = self.args['train_val_timeSteps']
        else: # self.mode == 'testing'
            self.timeSteps = self.args['test_timeSteps']


        self.trials = self.prepared_dataset.getDatasets(mode=self.mode)
        self.Idx_perDay = self.prepared_dataset.getDaysIdx(mode=self.mode)




    # method fro retreiving the indices of the trials for each day to be used in the DayBatchSampler
    def getDaysIdx(self):
        return self.Idx_perDay




    def __len__(self):
        return len(self.trials['neuralData'])



        
    def __getitem__(self, idx):

        if self.mode == 'training':
            # extracting the trial from the training dataset
            trial = {key: data[idx] for key, data in self.trials.items()}
            
            # extracting random snippets from the sentences
            extractSentenceSnippet(trial, self.timeSteps, self.args['directionality'])
            if self.args['constantOffsetSD'] > 0 or self.args['randomWalkSD'] > 0:
                # adding mean noise to the trial
                addMeanNoise(trial, self.args['constantOffsetSD'], self.args['randomWalkSD'],self.timeSteps)
            if self.args['whiteNoiseSD'] > 0:
                # adding white noise to the trial
                addWhiteNoise(trial, self.args['whiteNoiseSD'], self.timeSteps)

        elif self.mode == 'validation':
            trial = {key: data[idx] for key, data in self.trials.items()}
            # extracting random snippets from the sentences without adding noise
            extractSentenceSnippet(trial, self.timeSteps, self.args['directionality'])

        elif self.mode == 'testing':
            # no snippets are extracted, the whole trial is used and no noise is added
            trial = {key: data[idx] for key, data in self.trials.items()}
        
        else:
            raise ValueError('Mode not recognized')

        return trial


