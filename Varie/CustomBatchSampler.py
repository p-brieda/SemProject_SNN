import torch
from torch.utils.data import Sampler
import numpy as np
import math

# ALL-DAYS DATALOADER APPROACH
# TRAINING AND VALIDATION SAMPLER
# DayBatchSampler is a custom sampler that samples batches of trials from the dataset in order to have
# trials from the same day in the same batch. If the number of trials is not divisible by the batch size, the last
# batch will have fewer trials. The trials for each day are shuffled once before starting batch generation.
#
# For every batch creation a random day is selected with a probability proportional to its number of trials (relative
# to the total number of trials across all days). The sampler keeps track of the start index for each day and generates batches 
# until all trials have been used. If all trials of a day have been used, the sampler removes that day from the list of available days.

class CustomBatchSampler(Sampler):
    def __init__(self, Idx_perDay, batch_size):
        self.Idx_perDay = Idx_perDay
        self.batch_size = batch_size
        self.total_trials = sum(len(indices) for indices in self.Idx_perDay)
        # Calculate the probability for each day based on the number of trials
        self.probabilities = np.array([len(indices) / self.total_trials for indices in Idx_perDay])


    def __iter__(self):
        # Shuffle indices for each day once before starting batch generation
        shuffled_indices_per_day = [np.random.permutation(indices) for indices in self.Idx_perDay]
        start_indices = [0] * len(self.Idx_perDay)  # Start index for each day's batch generation

        # Continue until all indices have been yielded
        while any(start < len(indices) for start, indices in zip(start_indices, shuffled_indices_per_day)):
            # Randomly choose a day with remaining indices
            available_days = [i for i, start in enumerate(start_indices) if start < len(shuffled_indices_per_day[i])]

            # Choose a random day with probability proportional to the number of trials
            probs = self.probabilities[available_days] / np.sum(self.probabilities[available_days]) # normalization of remaining probabilities
            day = np.random.choice(available_days, p=probs)
            start = start_indices[day]
            day_indices = shuffled_indices_per_day[day]

            # Determine the end index for the current batch
            end = min(start + self.batch_size, len(day_indices))

            # Yield the current batch
            yield day_indices[start:end]

            # Update the start index for the next batch from this day
            start_indices[day] += self.batch_size


    def __len__(self):
        # the last batch may have fewer trials but it is taken into account into the length of the sampler
        total_batches = sum( (math.ceil(len(indices)/self.batch_size) for indices in self.Idx_perDay) )
        return total_batches