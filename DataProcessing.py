import os
from datetime import datetime
import random
import numpy as np
from util import prepareDataCubesForRNN, unfoldDataCube
from transforms import extractSentenceSnippet, addMeanNoise, addWhiteNoise
import torch
from torch.utils.data import Dataset




class DataProcessing(Dataset):
    def __init__(self, args, mode='training'):
        self.args = args
        self.mode = mode
        # selecting different amount of time steps for training/validation and testing
        if self.mode == 'training' or self.mode =='validation':
            self.timeSteps = self.args['train_val_timeSteps']
        else: # self.mode == 'testing'
            self.timeSteps = self.args['test_timeSteps']


        #count how many days of data are specified
        self.nDays = 0
        for t in range(30):
            if 'labelsFile_'+str(t) not in self.args.keys():
                self.nDays = t
                break

        # load all datasets, both for training and validation/test and the full dataset
        self.trials_train, self.trials_val, self.trainIdx_perDay, self.valIdx_perDay = self._loadAllDatasets()


        # setting random seed with argument or randomly, both for numpy and pytorch
        if self.args['seed']==-1:
            self.args['seed']=datetime.now().microsecond
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])



    # method fro retreiving the indices of the trials for each day to be used in the DayBatchSampler
    def getDaysIdx(self):
        if self.mode == 'training':
            return self.trainIdx_perDay
        else: # validation or testing
            return self.valIdx_perDay




    def __len__(self):
        if self.mode == 'training':
            return len(self.trials_train['neuralData'])
        else: # validation or testing
            return len(self.trials_val['neuralData'])
        


        
    def __getitem__(self, idx):

        if self.mode == 'training':
            # extracting the trial from the training dataset
            trial = {key: data[idx] for key, data in self.trials_train.items()}
            
            # extracting random snippets from the sentences
            extractSentenceSnippet(trial, self.timeSteps, self.args['directionality'])
            if self.args['constantOffsetSD'] > 0 or self.args['randomWalkSD'] > 0:
                # adding mean noise to the trial
                addMeanNoise(trial, self.args['constantOffsetSD'], self.args['randomWalkSD'],self.timeSteps)
            if self.args['whiteNoiseSD'] > 0:
                # adding white noise to the trial
                addWhiteNoise(trial, self.args['whiteNoiseSD'], self.timeSteps)

        elif self.mode == 'validation':
            trial = {key: data[idx] for key, data in self.trials_val.items()}
            # extracting random snippets from the sentences without adding noise
            extractSentenceSnippet(trial, self.timeSteps, self.args['directionality'])

        else: # self.mode == 'testing'
            # no snippets are extracted, the whole trial is used and no noise is added
            trial = {key: data[idx] for key, data in self.trials_val.items()}

        return trial




    def _loadAllDatasets(self):
    # Function that loads all the data from the different days and returns it in two dictionaries (one for training
    # and one for validation) in which each key contains a list of the corresponding data or each day.
    # The dictionaries have the following keys:
    #   'neuralData': neural data for the trial
    #   'targets': targets for the trial
    #   'errWeights': error weights for the trial
    #   'binsPerTrial': number of bins for the trial
    #   'dayIdx': index of the day in the dataset
    #
    # The function also returns two lists (one for training and one for validation) with the indices of the trials 
    # for each day to be used in the DayBatchSampler.

    # CHANGES: synthetic data loading has been deleted

        # initilaising the dictionaries and lists

        trials_train = {'neuralData':[],'targets':[],'errWeights':[],'binsPerTrial':[],'dayIdx':[]}
        trainIdx_perDay = []
        tot_train_trials = 0

        trials_val = {'neuralData':[],'targets':[],'errWeights':[],'binsPerTrial':[],'dayIdx':[]}
        valIdx_perDay = []
        tot_val_trials = 0

    
        for dayIdx in range(self.nDays):
            # preprocessing the data cubes
            neuralData, targets, errWeights, binsPerTrial, cvIdx = prepareDataCubesForRNN(self.args['sentencesFile_'+str(dayIdx)],
                                                                                        self.args['singleLettersFile_'+str(dayIdx)],
                                                                                        self.args['labelsFile_'+str(dayIdx)],
                                                                                        self.args['cvPartitionFile_'+str(dayIdx)],
                                                                                        self.args['sessionName_'+str(dayIdx)],
                                                                                        self.args['rnnBinSize'],
                                                                                        self.timeSteps,
                                                                                        self.mode == 'training' or self.mode == 'validation')
            
            num_trainTrials = len(cvIdx['trainIdx']) # number of training trials of the day
            num_valTrials = len(cvIdx['testIdx']) # number of validation trials of the day
            
            # adding the indexes of the training trials of the day to the list
            trainIdx_perDay.append(np.arange(tot_train_trials, tot_train_trials+num_trainTrials))
            tot_train_trials += num_trainTrials

            # adding the indexes of the validation trials of the day to the list
            valIdx_perDay.append(np.arange(tot_val_trials, tot_val_trials+num_valTrials))
            tot_val_trials += num_valTrials
            
            # unfoling the data cubes and adding the trials data to the dictionaries
            unfoldDataCube(trials_train, trials_val, neuralData, targets, errWeights, binsPerTrial, cvIdx, dayIdx)

        return trials_train, trials_val, trainIdx_perDay, valIdx_perDay
        


