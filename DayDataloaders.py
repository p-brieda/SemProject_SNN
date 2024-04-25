# SINGLE-DAY DATALOADER APPROACH

from PrepareDataSet import PrepareDataSet
from transforms import extractSentenceSnippet, addMeanNoise, addWhiteNoise
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Class for preparing the single-day dataset for one of the three modes: training, validation or testing

class DayDataProcessing(Dataset):
    def __init__(self, args, prepared_dataset, mode='training'):
        self.args = args
        self.mode = mode
        self.prepared_dataset = prepared_dataset
        
        # selecting different amount of time steps for training/validation and testing
        if self.mode == 'training' or self.mode =='validation':
            self.timeSteps = self.args['train_val_timeSteps']
        else: # self.mode == 'testing'
            self.timeSteps = self.args['test_timeSteps']

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
    def __init__(self, args, days, mode):
        self.datasets = []
        self.dataloaders = []
        self.viabledays = []

        if mode == 'training' or mode == 'validation':
            Shuffle = True
        else: Shuffle = False

        for day in days:
            prepared_dataset = PrepareDataSet(args, days=[day])
            self.datasets.append(DayDataProcessing(args, prepared_dataset, mode))

            if self.datasets[-1].isViableDay():
                self.viabledays.append(day)
                self.dataloaders.append(DataLoader(self.datasets[-1], batch_size=args['batchSize'], shuffle=Shuffle, num_workers=0))

    
    def getDataloaders(self):
        return self.dataloaders
    
    def getViableDays(self):
        return self.viabledays
    
    
        
        
