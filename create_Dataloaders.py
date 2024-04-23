from PrepareDataSet import PrepareDataSet
from DayDataProcessing import DayDataProcessing
from torch.utils.data import DataLoader

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
    
    