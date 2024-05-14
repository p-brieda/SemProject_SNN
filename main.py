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
    logging.info('Epochs strategy')
    logging.info(' ')



    # ---------- DATASET PREPARATION ----------
    # Check if the data has already been prepared
    manual_prep = input('Do you want to manual control over data preparation? (y/n) ') == 'y'

    if hyperparams['system'] == 'Linux':
        prepared_data_dir = '/scratch/sem24f8/dataset/'
    else:
        prepared_data_dir = 'dataset/'

    hyperparams['prepared_data_dir'] = prepared_data_dir

    if not os.path.exists(prepared_data_dir + 'prepared_data.pth') or (manual_prep and input('Do you want to recompute the prepared data? (y/n) ') == 'y'):
        dataprep_start = time.time()
        prepared_data = PrepareData(hyperparams)
        dataprep_end = time.time()
        print(f'Data preparation time: {dataprep_end - dataprep_start:.2f} s')
        torch.save(prepared_data, prepared_data_dir + 'prepared_data.pth')
        print('Data prepared')
        logging.info(f"Preparing data file from raw data")
    else:
        print('Loading prepared data from dir')
        logging.info(f"Loading prepared data from dir")
        prepared_data = torch.load(prepared_data_dir + 'prepared_data.pth')
        print('Data loaded')
        


  
    # loading the training dataset and creating a DataLoader
    Train_dataset = DataProcessing(hyperparams, prepared_data, mode='training')
    trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), hyperparams['batch_size'])
    train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=0)
    print('Training dataloader ready')
    logging.info(f"Training dataloaders ready")


    # loading the validation dataset and creating a DataLoader
    Val_dataset = DataProcessing(hyperparams, prepared_data, mode='validation')
    valDayBatch_Sampler = CustomBatchSampler(Val_dataset.getDaysIdx(), hyperparams['batch_size'])
    val_loader = DataLoader(Val_dataset, batch_sampler = valDayBatch_Sampler, num_workers=0)
    print('Validation dataloader ready')
    logging.info(f"Validation dataloaders ready")



    # ---------- MODEL CREATION ----------
    # Device selection
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU available')
        device = torch.device(hyperparams['device'])
    print(f'Device: {device}')
    logging.info(f"Using {device}")

    # Model creation
    model = RSNNet(hyperparams)
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs")
    #    model = nn.DataParallel(model)
    model.to(device)

    # Loss function
    criterion = SequenceLoss(hyperparams)

    
    num_batches_per_epoch_train = len(train_loader)
    num_batches_per_epoch_val = len(val_loader)
    epochs = hyperparams['epochs']
    tot_train_batches = num_batches_per_epoch_train * epochs
    logging.info(f"Number of training batches: {num_batches_per_epoch_train}")
    logging.info(f"Number of validation batches: {num_batches_per_epoch_val}")
    logging.info(f"Number of epochs: {epochs}")
    logging.info(f"Total training batches: {tot_train_batches}")
    logging.info(f"Batch size: {hyperparams['batch_size']}")    
    logging.info(f"Time steps: {hyperparams['train_val_timeSteps']}")
    logging.info(f"White noise: {hyperparams['whiteNoiseSD']}")
    if hyperparams['smoothInputs']: logging.info(f"Smoothing inputs")


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], betas= (0.9, 0.999), eps=1e-08, weight_decay=hyperparams['weight_decay'], amsgrad=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['learning_rate'], momentum=0.9, weight_decay=hyperparams['weight_decay'])
    logging.info(f"Optimizer: AdamW(lr={hyperparams['learning_rate']}, betas=(0.9, 0.999), eps=1e-08, weight_decay={hyperparams['weight_decay']}, amsgrad=False)")
    #logging.info(f"Optimizer: SGD(lr={hyperparams['learning_rate']}, momentum=0.9, weight_decay={hyperparams['weight_decay']})")



    # Scheduler 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: (1 - i/tot_train_batches))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    logging.info(f"Scheduler: LambdaLR(lr_lambda=lambda i: (1 - i/{tot_train_batches}))")
    logging.info(' ')
    
    

    # ---------- TRAINING AND VALIDATION ----------
    # Start timer
    training_start = time.time()
    print(f"Number of training batches/epoch: {num_batches_per_epoch_train}")

    trainloss_per_batch = []
    trainloss_per_epoch = []
    valloss_per_batch = []
    valloss_per_epoch = []

    trainacc_per_batch = []
    trainacc_per_epoch = []
    valacc_per_batch = []
    valacc_per_epoch = []

    train_sc_per_batch = []
    train_sc_per_epoch = []
    val_sc_per_batch = []
    val_sc_per_epoch = []

    for epoch in range(epochs):
        # Start epoch timer
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}")
        logging.info(f"Epoch: {epoch+1}")

        # Training epoch
        train_loss, train_acc = trainModel(model, train_loader , optimizer, scheduler, criterion, hyperparams, device)
        # Validation epoch
        val_loss, val_acc = validateModel(model, val_loader, criterion, hyperparams, device)


        epoch_end = time.time()
        print(f"Epoch time: {epoch_end - epoch_start:.2f} s ; Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        logging.info(f"Epoch time: {epoch_end - epoch_start:.2f} s")

        # Metrics saving
        trainloss_per_batch.append(train_loss)
        trainacc_per_batch.append(train_acc)
        valloss_per_batch.append(val_loss)
        valacc_per_batch.append(val_acc)
        

        avg_train_loss_epoch = np.sum(train_loss)/num_batches_per_epoch_train
        avg_train_acc_epoch = np.sum(train_acc)/num_batches_per_epoch_train
        avg_val_loss_epoch = np.sum(val_loss)/num_batches_per_epoch_val
        avg_val_acc_epoch = np.sum(val_acc)/num_batches_per_epoch_val


        trainloss_per_epoch.append(avg_train_loss_epoch)
        trainacc_per_epoch.append(avg_train_acc_epoch)
        valloss_per_epoch.append(avg_val_loss_epoch)
        valacc_per_epoch.append(avg_val_acc_epoch)
        

        # Print results of the epoch
        print(f"Train loss: {avg_train_loss_epoch:.4f} | Train accuracy: {avg_train_acc_epoch:.4f} | Val loss: {avg_val_loss_epoch:.4f} | Val accuracy: {avg_val_acc_epoch:.4f}")
        logging.info(f"Train loss: {avg_train_loss_epoch:.4f} | Train accuracy: {avg_train_acc_epoch:.4f} | Val loss: {avg_val_loss_epoch:.4f} | Val accuracy: {avg_val_acc_epoch:.4f}")
        logging.info(' ')

        if epoch % 10 == 0:
            # Save the metrics
            metrics = {'trainloss_per_batch': trainloss_per_batch, 'trainloss_per_epoch': trainloss_per_epoch,
                        'trainacc_per_batch': trainacc_per_batch, 'trainacc_per_epoch': trainacc_per_epoch,
                        'valloss_per_batch': valloss_per_batch, 'valloss_per_epoch': valloss_per_epoch,
                        'valacc_per_batch': valacc_per_batch, 'valacc_per_epoch': valacc_per_epoch}
            torch.save(metrics, f"trainOutputs/metrics_{hyperparams['id']}.pth")
            print('Metrics saved')
            logging.info('Metrics saved')


    training_end = time.time()
    print(f"Training time: {(training_end - training_start)/60:.2f} mins")
    logging.info(f"Training time: {(training_end - training_start)/60:.2f} mins")
    

    # Save the model
    torch.save(model, f"Model/model_{hyperparams['id']}.pth")
    print('Model saved')
    logging.info('Model saved')

    # Save the metrics
    metrics = {'trainloss_per_batch': trainloss_per_batch, 'trainloss_per_epoch': trainloss_per_epoch,
                'trainacc_per_batch': trainacc_per_batch, 'trainacc_per_epoch': trainacc_per_epoch,
                'valloss_per_batch': valloss_per_batch, 'valloss_per_epoch': valloss_per_epoch,
                'valacc_per_batch': valacc_per_batch, 'valacc_per_epoch': valacc_per_epoch}
    torch.save(metrics, f"trainOutputs/metrics_{hyperparams['id']}.pth")
    print('Final metrics saved')
    logging.info('Final metrics saved')

    
