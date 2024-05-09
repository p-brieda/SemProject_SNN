import torch
import numpy as np
import scipy.io
import os
import sys
import time
from datetime import datetime
import logging

# Custom imports and torch imports
from util import getDefaultHyperparams, extractBatch, trainModel, validateModel
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareData import PrepareData
from torch.utils.data import DataLoader
from network import Net, RSNNet
from SequenceLoss import SequenceLoss
import torch.nn as nn

# RayTune imports
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray_config import ray_config_dict


def main():

    # SET AN EXPERIMENT NAME
    EXPERIMENT_NAME = 'ANN_JL_new_params_6'
    hyperparams = getDefaultHyperparams()
    hyperparams['batch_size'] = 20
    hyperparams['train_val_timeSteps'] = 1200
    
    hyperparams['n_channels'] = 192
    hyperparams['n_outputs'] = 32

    # torch.set_num_threads = 3
    config_name = "baseline"
    ray_config = ray_config_dict(hyperparams, EXPERIMENT_NAME, config_name)

    optuna_search = OptunaSearch()
    ashas_scheduler = ASHAScheduler(grace_period=5, reduction_factor=5)

    # CONFIGURE RAY TUNE
    # num_samples: when there is grid search, the number of samples is the number of full exploration of the space
    #              When there is only random function for tune.choice, it indicates the samples into the space.
    analysis = tune.run(train_tune_parallel,
                        config=ray_config,
                        resources_per_trial={'cpu': 1, 'gpu':1}, 
                        # resources_per_trial={'cpu': 1, 'gpu':0.5},  # This is default setup
                        max_concurrent_trials = 1,
                        num_samples = 1,
                        # search_alg=optuna_search,
                        #scheduler=ashas_scheduler, 
                        metric='eval_corr_mean',
                        local_dir = '/usr/scratch2/sassauna3/liaoj/snn/log',
                        mode='max',
                        name=EXPERIMENT_NAME + '_' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
                        )



def train_tune_parallel(config):

    hyperparams = config.pop('hyperparam')
    for key, value in config.items():
        hyperparams[key] = value

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
        prepared_data = torch.load(prepared_data_dir + 'prepared_data.pth')
        print('Data loaded')
        logging.info(f"Loading prepared data from dir")


  
    # loading the training dataset and creating a DataLoader
    Train_dataset = DataProcessing(hyperparams, prepared_data, mode='training')
    trainDayBatch_Sampler = CustomBatchSampler(Train_dataset.getDaysIdx(), hyperparams['batch_size'])
    train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=2)
    print('Training dataloader ready')
    logging.info(f"Training dataloaders ready")


    # loading the validation dataset and creating a DataLoader
    Val_dataset = DataProcessing(hyperparams, prepared_data, mode='validation')
    valDayBatch_Sampler = CustomBatchSampler(Val_dataset.getDaysIdx(), hyperparams['batch_size'])
    val_loader = DataLoader(Val_dataset, batch_sampler = valDayBatch_Sampler, num_workers=2)
    print('Validation dataloader ready')
    logging.info(f"Validation dataloaders ready")


    # loading the testing dataset and creating a DataLoader for each day
    Test_finite_loader = create_Dataloaders(manual_prep, hyperparams, days=np.arange(10), mode='testing')
    test_loaders = Test_finite_loader.getDataloaders()
    viable_test_days = Test_finite_loader.getViableDays()
    print('Testing dataloaders ready')
    logging.info(f"Testing dataloaders ready")
    logging.info(' ')




    # ---------- MODEL CREATION ----------
    # Device selection
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU available')
        device = torch.device('cuda:0')
    print(f'Device: {device}')
    logging.info(f"Using {device}")

    # Model creation
    model = Net(hyperparams)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Loss function
    criterion = SequenceLoss(hyperparams)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], 
                                betas= (0.9, 0.999), eps=1e-08, 
                                weight_decay=hyperparams['weight_decay'], amsgrad=False)
    logging.info(f"Optimizer: AdamW(lr={hyperparams['learning_rate']}, betas=(0.9, 0.999), eps=1e-08, weight_decay={hyperparams['weight_decay']}, amsgrad=False)")

    
    num_batches_per_epoch_train = len(train_loader)
    num_batches_per_epoch_val = len(val_loader)
    epochs = hyperparams['epochs']
    logging.info(f"Number of training batches: {num_batches_per_epoch_train}")
    logging.info(f"Number of validation batches: {num_batches_per_epoch_val}")
    logging.info(f"Number of epochs: {epochs}")

    # Scheduler 
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: (1 - i/epochs))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    logging.info(f"Scheduler: StepLR(step_size=10, gamma=0.1)")
    logging.info(' ')
    
    

    # ---------- TRAINING AND VALIDATION ----------
    # Start timer
    training_start = time.time()
    print(f"Number of training batches/epoch: {num_batches_per_epoch_train}")

    trainloss_per_batch = []
    trainloss_per_epoch = []

    valloss_per_batch = []
    valloss_per_epoch = []

    valacc_per_batch = []
    valacc_per_epoch = []

    for epoch in range(epochs):
        # Start epoch timer
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}")
        logging.info(f"Epoch: {epoch+1}")

        # Training epoch
        train_loss = trainModel(model, train_loader , optimizer, scheduler, criterion, hyperparams, device)
        # Validation epoch
        val_loss, val_acc = validateModel(model, val_loader, criterion, hyperparams, device)


        epoch_end = time.time()
        print(f"Epoch time: {epoch_end - epoch_start:.2f} s ; Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        logging.info(f"Epoch time: {epoch_end - epoch_start:.2f} s")

        trainloss_per_batch.append(train_loss)
        valloss_per_batch.append(val_loss)
        valacc_per_batch.append(val_acc)

        avg_train_loss_epoch = np.sum(train_loss)/num_batches_per_epoch_train
        avg_val_loss_epoch = np.sum(val_loss)/num_batches_per_epoch_val
        avg_val_acc_epoch = np.sum(val_acc)/num_batches_per_epoch_val

        trainloss_per_epoch.append(avg_train_loss_epoch)
        valloss_per_epoch.append(avg_val_loss_epoch)
        valacc_per_epoch.append(avg_val_acc_epoch)

        # Print results of the epoch
        print(f"Train loss: {avg_train_loss_epoch:.4f} | Validation loss: {avg_val_loss_epoch:.4f} | Validation accuracy: {np.sum(val_acc)/num_batches_per_epoch_val:.4f}")
        logging.info(f"Train loss: {avg_train_loss_epoch:.4f} | Validation loss: {avg_val_loss_epoch:.4f} | Validation accuracy: {avg_val_acc_epoch:.4f}")
        logging.info(' ')

    training_end = time.time()
    print(f"Training time: {(training_end - training_start)/60:.2f} mins")
    logging.info(f"Training time: {(training_end - training_start)/60:.2f} mins")
    

    # Save the model
    torch.save(model.state_dict(), 'Model/model.pth')
    print('Model saved')
    logging.info('Model saved')

    # Save the metrics
    metrics = {'trainloss_per_batch': trainloss_per_batch, 'trainloss_per_epoch': trainloss_per_epoch,
                'valloss_per_batch': valloss_per_batch, 'valloss_per_epoch': valloss_per_epoch,
                'valacc_per_batch': valacc_per_batch, 'valacc_per_epoch': valacc_per_epoch}
    torch.save(metrics, 'Model/metrics.pth')


if __name__ == '__main__':
    main()
