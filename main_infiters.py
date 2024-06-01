import torch
import numpy as np
import scipy.io
import os
import logging
import sys
import time
from util import getDefaultHyperparams, extractBatch, trainModel_Inf, validateModel, neuron_hist_plot, TrainPlot, modelComplexity
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareData import PrepareData
from torch.utils.data import DataLoader
from network import Net, RSNNet
from RNN_network import RNN
from SequenceLoss import SequenceLoss
import torch.nn as nn




if __name__ == '__main__':
    hyperparams = getDefaultHyperparams()
    
    hyperparams['n_channels'] = 192
    hyperparams['n_outputs'] = 32

    #prepare logger
    logging.basicConfig(filename=hyperparams['output_report'],
                                filemode='a',
                                format='%(asctime)s,%(msecs)d --- %(message)s',
                                datefmt='%H:%M:%S',
                                level=15)
    

    hyperparams['id'] = np.random.randint(100000,1000000)
    logging.info(' ')
    logging.info("=========================================================================")
    logging.info(f"New run started with id {hyperparams['id']}")
    print(f"ID: {hyperparams['id']}", end=' ')
    logging.info(f"ID: {hyperparams['id']}")
    logging.info('Infinite iterators strategy')



    # ---------- DATASET PREPARATION ----------
    # Check if the data has already been prepared
    #manual_prep = input('Do you want to manual control over data preparation? (y/n) ') == 'y'
    manual_prep = False

    if hyperparams['system'] == 'Linux':
        prepared_data_dir = '/scratch/sem24f8/dataset/'
    else:
        prepared_data_dir = 'dataset/'

    hyperparams['prepared_data_dir'] = prepared_data_dir

    # loading the training dataset and creating a Dataloader for each day
    Train_finite_loader = create_Dataloaders(manual_prep, hyperparams, days=np.arange(10), mode='training')
    train_loaders = Train_finite_loader.getDataloaders()
    viable_train_days = Train_finite_loader.getViableDays()
    print(f"Viable training days: {viable_train_days}")
    logging.info(f"Viable training days: {viable_train_days}")
    # create infinite iterators
    train_inf_iterators = DayInfiniteIterators(train_loaders)
    print('Training dataloaders ready')
    logging.info(f"Training dataloaders ready")
    logging.info(' ')


    # loading the validation dataset and creating a DataLoader for each day
    Val_finite_loader = create_Dataloaders(manual_prep, hyperparams, days=np.arange(10), mode='validation')
    val_loaders = Val_finite_loader.getDataloaders()
    viable_val_days = Val_finite_loader.getViableDays()
    print(f"Viable validation days: {viable_val_days}")
    logging.info(f"Viable validation days: {viable_val_days}")
    # create infinite iterators
    val_inf_iterators = DayInfiniteIterators(val_loaders)
    print('Validation dataloaders ready')
    logging.info(f"Validation dataloaders ready")
    logging.info('')



    # ---------- MODEL CREATION ----------
    # Device selection
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU available')
        device = torch.device(hyperparams['device'])
    print(f'Device: {device}')
    logging.info(f"Using {device}")

    # Model creation
    if hyperparams['network_type'] == 'RSNN':
        model = RSNNet(hyperparams)
    elif hyperparams['network_type'] == 'RNN':
        model = RNN(hyperparams)
    model.to(device)

    tot_train_batches = 55
    logging.info(f"Total training batches: {tot_train_batches}")
    logging.info(f"Batch size: {hyperparams['batch_size']}")
    logging.info(f"Time steps: {hyperparams['train_val_timeSteps']}")
    logging.info(f"White noise: {hyperparams['whiteNoiseSD']}")

    # Loss function
    criterion = SequenceLoss(hyperparams)

    # Optimizer
    if hyperparams['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['lr'], 
                                    betas= (0.9, 0.999), eps=hyperparams['eps'], 
                                    weight_decay=hyperparams['weight_decay'], amsgrad=False)
        logging.info(f"Optimizer: AdamW(lr={hyperparams['lr']}, betas=(0.9, 0.999), eps={hyperparams['eps']}, weight_decay={hyperparams['weight_decay']}, amsgrad=False)")

    elif hyperparams['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], 
                                     betas=(0.9, 0.999), eps=hyperparams['eps'], 
                                     weight_decay=hyperparams['weight_decay'], amsgrad=False)
        logging.info(f"Optimizer: Adam(lr={hyperparams['lr']}, betas=(0.9, 0.999), eps={hyperparams['eps']}, weight_decay={hyperparams['weight_decay']}, amsgrad=False)")


    # Scheduler 
    if hyperparams['scheduler'] == 'LambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: (1 - i/20000))
        logging.info(f"Scheduler: LambdaLR(lr_lambda=lambda i: (1 - i/{20000}))")

    elif hyperparams['scheduler'] == 'StepLR':
        # since the scheduler is inside the epoch loop, the actual step_size is in terms of batches
        step_size = hyperparams['step_size']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=hyperparams['gamma'])
        logging.info(f"Scheduler: StepLR(step_size={hyperparams['step_size']}, gamma={hyperparams['gamma']})")

    elif hyperparams['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=hyperparams['gamma'], patience=hyperparams['patience'], 
                                                               threshold=hyperparams['threshold'], threshold_mode='abs')
        logging.info(f"Scheduler: ReduceLROnPlateau(mode='min', factor={hyperparams['gamma']}, patience={hyperparams['patience']}, threshold={hyperparams['threshold']}, threshold_mode='abs')")
    
    logging.info(' ')
    
    

    # ---------- TRAINING AND VALIDATION ----------
    # Start timer
    training_start = time.time()

    trainloss_per_batch = []
    valloss_per_batch = []
    trainacc_per_batch = []
    valacc_per_batch = []


    # Training process
    train_loss, train_acc, val_loss, val_acc = trainModel_Inf(tot_train_batches, model, 
                                       train_inf_iterators, viable_train_days,
                                       val_inf_iterators, viable_val_days,
                                       optimizer, scheduler, criterion, hyperparams, device)
    

    training_end = time.time()
    print(f"Training time: {(training_end - training_start)/60:.2f} mins")
    logging.info(f"Training time: {(training_end - training_start)/60:.2f} mins")

    # save metrics
    metrics = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
    torch.save(metrics, f"trainOutputs/metrics_inf_{hyperparams['id']}.pth")
    

    # Save the model
    torch.save(model.state_dict(), f"Model/model_inf_{hyperparams['id']}.pth")
    print('Model saved')
    logging.info('Model saved')

   