import torch
from torch.utils.data import Sampler
import numpy as np
import math

# DayBatchSampler is a custom sampler that samples batches of trials from the dataset in order to have
# trials from the same day in the same batch. Ff the number of trials is not divisible by the batch size,
# the last batch will have fewer trials.
class DayBatchSampler(Sampler):
    def __init__(self, Idx_perDay, batch_size):
        self.Idx_perDay = Idx_perDay
        self.batch_size = batch_size


    def __iter__(self):
        day_order = torch.randperm(len(self.Idx_perDay))
        for day in day_order:
            day_idx = self.Idx_perDay[day]
            np.random.shuffle(day_idx)
            for i in range(0, len(day_idx), self.batch_size):
                yield day_idx[i:i+self.batch_size]

    def __len__(self):
        # the last batch may have fewer trials but it is taken into account intio the lenght of the sampler
        total_batches = sum( (math.ceil(len(indices)/self.batch_size) for indices in self.Idx_perDay) )
        return total_batches