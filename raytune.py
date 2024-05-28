import torch
import numpy as np
import scipy.io
import os
import sys
import time
from datetime import datetime
import logging
import pickle

# Custom imports and torch imports
from util import getDefaultHyperparams, extractBatch, trainModel, validateModel, neuron_hist_plot, TrainPlot, modelComplexity
from SingleDataloader import DataProcessing, CustomBatchSampler, TestBatchSampler
from DayDataloaders import create_Dataloaders, DayInfiniteIterators
from PrepareData import PrepareData
from torch.utils.data import DataLoader
from network import Net, RSNNet
from RNN_network import RNN
from SequenceLoss import SequenceLoss
import torch.nn as nn

# RayTune imports
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray_config import ray_config_dict


def main():

    # SET AN EXPERIMENT NAME
    EXPERIMENT_NAME = "ASHA_combo"
    hyperparams = getDefaultHyperparams()

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # torch.set_num_threads = 3
    config_name = "ASHA_combined"
    ray_config = ray_config_dict(hyperparams, config_name)


    #optuna_search = OptunaSearch()
    asha_scheduler = ASHAScheduler(time_attr='training_iteration', metric='val_acc', mode='max', max_t=hyperparams['epochs'], grace_period=500, reduction_factor=2)

    # CONFIGURE RAY TUNE
    # num_samples: when there is grid search, the number of samples is the number of full exploration of the space
    #              When there is only random function for tune.choice, it indicates the samples into the space.
    local_dir_path = 'C:/Users/pietr/OneDrive/Documenti/PIETRO/ETH/SS24/Semester_project/Files/Raytune/'
    if hyperparams['system'] == 'Linux':
        local_dir_path = '/home/sem24f8/Semester_project/SNN_Project/Files/Raytune/'
    
    reporter = tune.CLIReporter(
        metric_columns=["ID","epoch", "t_epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        max_report_frequency=60
    )

    # if using ASHA, please uncomment the scheduler parameter and comment the metric and mode parameters
    analysis = tune.run(train_tune_parallel,
                        config=ray_config,
                        resources_per_trial={'cpu': 2, 'gpu':0.25}, 
                        max_concurrent_trials = 4,
                        num_samples = 1,
                        progress_reporter=reporter,
                        # search_alg=optuna_search,
                        scheduler=asha_scheduler, 
                        #metric='val_acc',
                        local_dir = local_dir_path + 'log',
                        #mode='max',
                        name=EXPERIMENT_NAME + '_' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
                        )
    
    results_filename = local_dir_path + 'Trial_' + str(round(time.time())) + '.pickle'
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, 'wb') as f:
        pickle.dump(analysis, f, protocol=pickle.HIGHEST_PROTOCOL)
    return analysis 



def train_tune_parallel(config):

    hyperparams = config.pop('hyperparam')
    for key, value in config.items():
        hyperparams[key] = value

    # ---------- LOGGING ----------
    hyperparams['id'] = np.random.randint(100000,1000000)

    #prepare logger
    logging.basicConfig(filename=hyperparams['output_report'],
                                filemode='a',
                                format='%(asctime)s,%(msecs)d --- %(message)s',
                                datefmt='%H:%M:%S',
                                level=15)
    
    logging.info(' ')
    logging.info("=========================================================================")
    logging.info(f"New run started with id {hyperparams['id']}")
    print(f"ID: {hyperparams['id']}", end=' ')
    logging.info('Epochs strategy')
    logging.info(' ')



    # ---------- DATASET PREPARATION ----------
    manual_prep = False
    prepared_data_dir = hyperparams['prepared_data_dir']
    
    if not os.path.exists(prepared_data_dir + 'prepared_data.pth') or (manual_prep and input('Do you want to recompute the prepared data? (y/n) ') == 'y'):
        print('Preparing data')
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
    train_loader = DataLoader(Train_dataset, batch_sampler = trainDayBatch_Sampler , num_workers=1)
    print('Training dataloader ready')
    logging.info(f"Training dataloaders ready")


    # loading the validation dataset and creating a DataLoader
    Val_dataset = DataProcessing(hyperparams, prepared_data, mode='validation')
    valDayBatch_Sampler = CustomBatchSampler(Val_dataset.getDaysIdx(), hyperparams['batch_size'], fill_batch = False)
    val_loader = DataLoader(Val_dataset, batch_sampler = valDayBatch_Sampler, num_workers=1)
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
    if hyperparams['network_type'] == 'RSNN':
        model = RSNNet(hyperparams)
    elif hyperparams['network_type'] == 'RNN':
        model = RNN(hyperparams)
    model.to(device)


    # Loss function
    criterion = SequenceLoss(hyperparams)

    
    num_batches_per_epoch_train = len(train_loader)
    num_batches_per_epoch_val = len(val_loader)
    epochs = hyperparams['epochs']
    tot_train_batches = num_batches_per_epoch_train * epochs
    logging.info(f"Number of training batches / epoch: {num_batches_per_epoch_train}")
    logging.info(f"Number of validation batches / epoch: {num_batches_per_epoch_val}")
    logging.info(f"Number of epochs: {epochs}")
    logging.info(f"Total training batches: {tot_train_batches}")


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
        step_size = hyperparams['step_size'] * num_batches_per_epoch_train
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
    trainloss_per_epoch = []

    valloss_per_batch = []
    valloss_per_epoch = []

    trainacc_per_batch = []
    trainacc_per_epoch = []

    valacc_per_batch = []
    valacc_per_epoch = []

    lr_per_epoch = []

    for epoch in range(epochs):
        # Start epoch timer
        epoch_start = time.time()
        #print(f"\nEpoch {epoch+1}")
        logging.info(f"{hyperparams['id']} - Epoch: {epoch+1}")

        # Training epoch
        train_loss, train_acc = trainModel(model, train_loader , optimizer, scheduler, criterion, hyperparams, device)
        # Validation epoch
        val_loss, val_acc = validateModel(model, val_loader, criterion, hyperparams, device)


        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        #print(f"Epoch time: {epoch_time:.2f} s ; Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        #logging.info(f"Epoch time: {epoch_time:.2f} s ; Learning rate: {scheduler.get_last_lr()[0]:.6f}")

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

        lr_per_epoch.append(scheduler.get_last_lr()[0])

    
        # Print results of the epoch
        #print(f"Training loss: {avg_train_loss_epoch:.4f} | Training accuracy: {avg_train_acc_epoch:.4f} | Validation loss: {avg_val_loss_epoch:.4f} | Validation accuracy: {avg_val_acc_epoch:.4f}")
        #logging.info(f"Training loss: {avg_train_loss_epoch:.4f} | Training accuracy: {avg_train_acc_epoch:.4f} | Validation loss: {avg_val_loss_epoch:.4f} | Validation accuracy: {avg_val_acc_epoch:.4f}")
        logging.info(' ')

        if epoch % 20 == 0:
            # Save the metrics
            metrics = {'trainloss_per_batch': trainloss_per_batch, 'trainloss_per_epoch': trainloss_per_epoch,
                        'trainacc_per_batch': trainacc_per_batch, 'trainacc_per_epoch': trainacc_per_epoch,
                        'valloss_per_batch': valloss_per_batch, 'valloss_per_epoch': valloss_per_epoch,
                        'valacc_per_batch': valacc_per_batch, 'valacc_per_epoch': valacc_per_epoch,
                        'lr': lr_per_epoch}
            
            torch.save(metrics, f"{hyperparams['results_dir']}metrics_{hyperparams['id']}.pth")
            print('Metrics saved')
            logging.info('Metrics saved')

            # Save the model
            torch.save(model, f"{hyperparams['save_model_dir']}model_{hyperparams['id']}.pth")
            print('Model saved')
            logging.info('Model saved')

        # save metrics in tune report
        tune.report(
            ID=hyperparams['id'],
            epoch=epoch+1,
            t_epoch = epoch_time,
            train_loss=avg_train_loss_epoch,
            train_acc=avg_train_acc_epoch,
            val_loss=avg_val_loss_epoch,
            val_acc=avg_val_acc_epoch,
            lr=scheduler.get_last_lr()[0]
            )



    training_end = time.time()
    print(f"Training time: {(training_end - training_start)/60:.2f} mins")
    logging.info(f"Training time: {(training_end - training_start)/60:.2f} mins")


    # ---------- NEURON HISTOGRAM PLOT ----------
    if hyperparams['network_type'] !='RNN':
        neuron_hist_plot(model, hyperparams)
    

    # ---------- SAVE MODEL AND METRICS ----------
    # Save the model
    torch.save(model, f"{hyperparams['save_model_dir']}model_{hyperparams['id']}.pth")
    print('Final model saved')
    logging.info('Final model saved')

    # Save the metrics
    metrics = {'trainloss_per_batch': trainloss_per_batch, 'trainloss_per_epoch': trainloss_per_epoch,
                        'trainacc_per_batch': trainacc_per_batch, 'trainacc_per_epoch': trainacc_per_epoch,
                        'valloss_per_batch': valloss_per_batch, 'valloss_per_epoch': valloss_per_epoch,
                        'valacc_per_batch': valacc_per_batch, 'valacc_per_epoch': valacc_per_epoch,
                        'lr': lr_per_epoch}
    torch.save(metrics, f"{hyperparams['results_dir']}metrics_{hyperparams['id']}.pth")
    print('Final metrics saved')
    logging.info('Final metrics saved')

    # save metrics
    with open(f"{hyperparams['results_dir']}results_{hyperparams['id']}.pickle", 'wb') as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save hyperparam
    with open(f"{hyperparams['results_dir']}hyperparam_{hyperparams['id']}.pickle", 'wb') as f:
        pickle.dump(hyperparams, f, protocol=pickle.HIGHEST_PROTOCOL)


    # ---------- TRAINING PLOT ----------
    TrainPlot(metrics, hyperparams)

    # ----------MODEL COMPLEXITY ----------
    if hyperparams['network_type'] != 'RNN':
        MACs, ACs = modelComplexity(hyperparams)
        print(f"{hyperparams['id']} --- MACs: {np.ceil(MACs)} ; ACs: {np.ceil(ACs)}")
        logging.info(f"{hyperparams['id']} --- MACs: {np.ceil(MACs)} ; ACs: {np.ceil(ACs)}")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
