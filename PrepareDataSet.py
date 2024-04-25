import os
from datetime import datetime
import random
import numpy as np
from util import prepareDataCubesForRNN, unfoldDataCube
from torch.utils.data import Dataset


# Class for preparing the trials and corrresponding indexes for training, validation and testing. The trials and inexes
# will be used to create the datasets and dataloaders

class PrepareDataSet:
    def __init__(self, args, days=None):
        self.args = args

        # if not specified count how many days of data
        if days == None:
            nDays = 0
            for t in range(30):
                if 'labelsFile_'+str(t) not in self.args.keys():
                    nDays = t
                    break
            self.Days = np.arange(nDays)
        else:
            self.Days = days


        self.trials_train, self.trials_val, self.trainIdx_perDay, self.valIdx_perDay = self._loadAllDatasets('trainval')
        _ , self.trials_test, _ , self.testIdx_perDay = self._loadAllDatasets('testing')




    def _loadAllDatasets(self, mode):

        # selecting different amount of time steps for training/validation and testing
        if mode == 'trainval':
            timeSteps = self.args['train_val_timeSteps']
        else: # self.mode == 'testing'
            timeSteps = self.args['test_timeSteps']

        trials_train = {'neuralData':[],'targets':[],'errWeights':[],'binsPerTrial':[],'dayIdx':[]}
        trainIdx_perDay = []
        tot_train_trials = 0

        trials_val = {'neuralData':[],'targets':[],'errWeights':[],'binsPerTrial':[],'dayIdx':[]}
        valIdx_perDay = []
        tot_val_trials = 0

    
        for dayIdx in self.Days:
            # preprocessing the data cubes
            neuralData, targets, errWeights, binsPerTrial, cvIdx = prepareDataCubesForRNN(self.args['sentencesFile_'+str(dayIdx)],
                                                                                        self.args['singleLettersFile_'+str(dayIdx)],
                                                                                        self.args['labelsFile_'+str(dayIdx)],
                                                                                        self.args['cvPartitionFile_'+str(dayIdx)],
                                                                                        self.args['sessionName_'+str(dayIdx)],
                                                                                        self.args['rnnBinSize'],
                                                                                        timeSteps,
                                                                                        mode == 'trainval')
            
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
    

    

    def getDatasets(self, mode):
        if mode == 'training':
            return self.trials_train
        elif mode == 'validation':
            return self.trials_val
        elif mode == 'testing':
            return self.trials_test
        else:
            raise ValueError('Mode not recognized')



    def getDaysIdx(self, mode):
        if mode == 'training':
            return self.trainIdx_perDay
        elif mode == 'validation':
            return self.valIdx_perDay
        elif mode == 'testing':
            return self.testIdx_perDay
        else:
            raise ValueError('Mode not recognized')