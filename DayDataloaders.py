# SINGLE-DAY DATALOADER APPROACH

from PrepareDataSet import PrepareDataSet
from transforms import extractSentenceSnippet, addMeanNoise, addWhiteNoise
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Class for preparing the single-day dataset for one of the three modes: training, validation or testing

class DayDataProcessing(Dataset):
    def __init__(self, hyperparam, prepared_dataset, mode='training'):
        self.hyperparams = hyperparam
        self.mode = mode
        self.prepared_dataset = prepared_dataset
        
        # selecting different amount of time steps for training/validation and testing
        if self.mode == 'training' or self.mode =='validation':
            self.timeSteps = self.hyperparams['train_val_timeSteps']
        else: # self.mode == 'testing'
            self.timeSteps = self.hyperparams['test_timeSteps']

        self.trials = self.prepared_dataset.getDatasets(mode=self.mode)




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
        



# Class for creating the dataloaders for the training, validation and testing datasets for all the days

class create_Dataloaders:
    def __init__(self, hyperparam, days, mode):
        self.datasets = []
        self.dataloaders = []
        self.viabledays = []

        if mode == 'training' or mode == 'validation':
            Shuffle = True
        else: Shuffle = False

        for day in days:
            prepared_dataset = PrepareDataSet(hyperparam, days=[day])
            self.datasets.append(DayDataProcessing(hyperparam, prepared_dataset, mode))

            if self.datasets[-1].isViableDay():
                self.viabledays.append(day)
                self.dataloaders.append(DataLoader(self.datasets[-1], batch_size=hyperparam['batch_size'], shuffle=Shuffle, num_workers=0))

    
    def getDataloaders(self):
        return self.dataloaders
    
    def getViableDays(self):
        return self.viabledays
    
    
        
        
