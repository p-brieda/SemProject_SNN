import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import time
from util import getDefaultHyperparams, extractBatch, trainModel, validateModel
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareData import PrepareData
from torch.utils.data import DataLoader
from network import Net, RSNNet
from SequenceLoss import SequenceLoss

#if __name__ == '__main__':

hyperparams = getDefaultHyperparams()
hyperparams['batch_size'] = 20
hyperparams['train_val_timeSteps'] = 1200



# ---------- DATASET PREPARATION ----------
prepared_data = PrepareData(hyperparams)

# loading the training dataset and creating a DataLoader
Train_dataset = DataProcessing(hyperparams, prepared_data, mode='training')
trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), hyperparams['batch_size'])
train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=2)
print('Training data loaded')

# loading the validation dataset and creating a DataLoader
Val_dataset = DataProcessing(hyperparams, prepared_data, mode='validation')
valDayBatch_Sampler = CustomBatchSampler(Val_dataset.getDaysIdx(), hyperparams['batch_size'])
val_loader = DataLoader(Val_dataset, batch_sampler = valDayBatch_Sampler, num_workers=2)
print('Validation data loaded')

# loading the testing dataset and creating a DataLoaders for each day
Test_finite_loader = create_Dataloaders(hyperparams, days=np.arange(10), mode='testing')
test_loaders = Test_finite_loader.getDataloaders()
viable_test_days = Test_finite_loader.getViableDays()



# ---------- MODEL CREATION ----------
# Device selection
device = torch.device('cpu')
if torch.cuda.is_available():
    print('GPU available')
    device = torch.device('cuda')

# Model creation
model = Net(hyperparams)
model.to(device)

# Loss function
criterion = SequenceLoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], 
                              betas= (0.9, 0.999), eps=1e-08, 
                              weight_decay=hyperparams['weight_decay'], amsgrad=False)

# ---------- TRAINING AND VALIDATION ----------
# Start timer
training_start = time.time()
epochs = hyperparams['epochs']

for epoch in epochs:
    # Start epoch timer
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}")

    # Training epoch
    train_loss = trainModel(model, train_loader, criterion, optimizer, device, hyperparams)
    # Validation epoch
    val_loss, val_acc = validateModel(model, val_loader, criterion, device, hyperparams)

    epoch_end = time.time()
    print(f"Epoch time: {epoch_end - epoch_start:.2f} s")
    # Print results of the epoch
    print(f"Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")


training_end = time.time()
print('Training time: ', training_end - training_start)
   


    
    
    
    
    
    
    
    
    
    
# INFINITE ITERATOR VERSION
'''
Finite_loader = create_Dataloaders(hyperparams, days=np.arange(10), mode='training')
train_loaders = Finite_loader.getDataloaders()
viable_train_days = Finite_loader.getViableDays()
train_InfIterators = DayInfiniteIterators(train_loaders)

strategy2_start = time.time()

tot_batches = 30
for k in range(tot_batches):
    dayIdx = np.random.choice(viable_train_days)
    next_iter = train_InfIterators.getNextIter(dayIdx)
    #print(k)
    print(next_iter['neuralData'].shape)
    print(next_iter['dayIdx'])

strategy2_end = time.time()
print('Strategy 2 time: ', strategy2_end - strategy2_start)
'''

    


# %%
