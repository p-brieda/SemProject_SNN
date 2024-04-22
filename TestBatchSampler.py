import torch
from torch.utils.data import Sampler
import numpy as np
import math

class DayBatchSampler(Sampler):
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
        total_batches = sum( (math.ceil(len(indices)/self.batch_size) for indices in self.Idx_perDay) )
        return total_batches
            






    