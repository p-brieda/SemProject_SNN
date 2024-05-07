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



if __name__ == '__main__':

    hyperparams = getDefaultHyperparams()
    hyperparams['batch_size'] = 20
    hyperparams['train_val_timeSteps'] = 1200
    hyperparams['n_channels'] = 192
    hyperparams['n_outputs'] = 32



    # ---------- DATASET PREPARATION ----------
    # Check if the data has already been prepared
    manual_prep = input('Do you want to manual control over data preparation? (y/n) ') == 'y'

    if not os.path.exists('dataset\\prepared_data.pth') or (manual_prep and input('Do you want to recompute the prepared data? (y/n) ') == 'y'):
        dataprep_start = time.time()
        prepared_data = PrepareData(hyperparams)
        dataprep_end = time.time()
        print(f'Data preparation time: {dataprep_end - dataprep_start:.2f} s')
        torch.save(prepared_data, 'dataset\\prepared_data.pth')
        print('Data prepared')
    else:
        prepared_data = torch.load('dataset\\prepared_data.pth')
        print('Data loaded')


  
    # loading the training dataset and creating a DataLoader
    Train_dataset = DataProcessing(hyperparams, prepared_data, mode='training')
    trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), hyperparams['batch_size'])
    train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=2)
    print('Training dataloader ready')


    # loading the validation dataset and creating a DataLoader
    Val_dataset = DataProcessing(hyperparams, prepared_data, mode='validation')
    valDayBatch_Sampler = CustomBatchSampler(Val_dataset.getDaysIdx(), hyperparams['batch_size'])
    val_loader = DataLoader(Val_dataset, batch_sampler = valDayBatch_Sampler, num_workers=2)
    print('Validation dataloader ready')


    # loading the testing dataset and creating a DataLoader for each day
    Test_finite_loader = create_Dataloaders(manual_prep, hyperparams, days=np.arange(10), mode='testing')
    test_loaders = Test_finite_loader.getDataloaders()
    viable_test_days = Test_finite_loader.getViableDays()
    print('Testing dataloaders ready')




    # ---------- MODEL CREATION ----------
    # Device selection
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU available')
        device = torch.device('cuda:0')
    print(f'Device: {device}')

    # Model creation
    model = Net(hyperparams)
    model.to(device)

    # Loss function
    criterion = SequenceLoss(hyperparams)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], 
                                betas= (0.9, 0.999), eps=1e-08, 
                                weight_decay=hyperparams['weight_decay'], amsgrad=False)
    
    # Scheduler
    num_batches_per_epoch = len(train_loader)
    epochs = hyperparams['epochs']
    tot_batches = epochs * num_batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: (1 - i/tot_batches))
    
    

    # ---------- TRAINING AND VALIDATION ----------
    # Start timer
    training_start = time.time()
    print(f"Number of training batches/epoch: {num_batches_per_epoch}")

    for epoch in range(epochs):
        # Start epoch timer
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}")

        # Training epoch
        train_loss = trainModel(model, train_loader , optimizer, scheduler, criterion, hyperparams, device)
        # Validation epoch
        val_loss, val_acc = validateModel(model, val_loader, criterion, hyperparams, device)

        epoch_end = time.time()
        print(f"Epoch time: {epoch_end - epoch_start:.2f} s")
        # Print results of the epoch
        print(f"Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

    training_end = time.time()
    print('Training time: ', training_end - training_start)

    # Save the model
    torch.save(model.state_dict(), 'Model\\model.pth')
