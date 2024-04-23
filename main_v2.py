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
from PrepareDataSet import PrepareDataSet

if __name__ == '__main__':

    args = getDefaultArgs()
    args['batchSize'] = 30

    All_trials = PrepareDataSet(args)
    # loading the training dataset
    Training_trials = All_trials.getDatasets(mode='training')
    train_days = All_trials.getDaysIdx(mode='training')

    Val_dataset = All_trials.getDatasets(mode='validation')
    val_days = All_trials.getDaysIdx(mode='validation')

    Test_dataset = All_trials.getDatasets(mode='testing')
    test_days = All_trials.getDaysIdx(mode='testing')
