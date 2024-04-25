import torch
from torch.utils.data import Sampler
import numpy as np
import math

# ALL-DAYS DATALOADER APPROACH
# TESTING SAMPLER
# TestBatchSampler is a custom sampler that samples batches of trials from the validation dataset for inference.
# The days to be tested are specified in the 'days' argument. The sampler generates batches of trials from the specified days
# in a sequential manner, without shuffling the trials. If the number of trials is not divisible by the batch size, the last
# batch will have fewer trials.

class TestBatchSampler(Sampler):
    def __init__(self, Idx_perDay, days, batch_size):
        self.Idx_perDay = Idx_perDay
        self.batch_size = batch_size
        self.daysToTest = days

    def __iter__(self):
        start = 0
        for day in self.daysToTest:
            while start < len(self.Idx_perDay[day]):
                day_indices = self.Idx_perDay[day]
                end = min(start + self.batch_size, len(day_indices))

                yield day_indices[start:end]
                start = end

            start = 0

    def __len__(self):
        # the last batch may have fewer trials but it is taken into account into the length of the sampler
        total_batches = sum( (math.ceil(len(indices)/self.batch_size) for indices in self.Idx_perDay) )
        return total_batches
            






    