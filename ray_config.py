#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on May 16 2023
# @author: liaoj


from ray import tune
import numpy as np


def ray_config_dict(hyperparam, config_name):
    case = {

        "baseline": {
            "seed": tune.randint(1,10000),
            "network_type": "SNN",
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "dropout": 0.0,
            "batchnorm": "tdBN",
            "Vth_trainable": False,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "RSNN_combined": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "dropout": 0.0,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },



        
        "RNN_baseline": {
            "seed": tune.randint(1,10000),
            "network_type": "RNN",
            "optimizer": "Adam",
            "lr": 0.01,
            "weight_decay": 0.00001,
            "epochs": 2400,
            "scheduler": "LambdaLR",
            "constantOffsetSD": 0.6,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.2,
            "hyperparam": hyperparam
        },



        "ASHA_combined": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            #"weight_decay": tune.grid_search([0.00001, 0.0001, 0.001]),
            "dropout": tune.grid_search([0.0, 0.3]),
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },



        "ASHA_noise": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": 0.0,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": tune.grid_search([0.5]),
            "randomWalkSD": 0.02,
            #"randomWalkSD": tune.grid_search([0.02]),
            "whiteNoiseSD": tune.grid_search([1.0]),
            "hyperparam": hyperparam
        },



        "ASHA_dropout": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": tune.grid_search([0.0, 0.2, 0.3, 0.4]),
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "PBT_combined": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "dropout": 0.0,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": tune.grid_search([0.3, 0.5]),
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },
        


        "nospike_search": {
            "seed": tune.randint(1,10000),
            "epochs": 1200,
            "network_type": "RSNN",
            "layers": 3,
            "last_nospike": tune.grid_search([False]),
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": 0.0,
            #"batchnorm": tune.grid_search(['none', 'tdBN']),
            #"recurrent_batchnorm": True,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "GaussSmooth_search": {
            "seed": tune.randint(1,10000),
            "epochs": 800,
            "network_type": "RSNN",
            "layers": 3,
            "smoothInputs": tune.grid_search([False, True]),
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": 0.0,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "SNN_hyper_search": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 3,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": 0.0,
            "noisy_threshold": 0.0,
            "batch_norm": 'tdBN',
            "recurrent_batchnorm": True,
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": 0.3,
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "architecture_search": {
            "seed": tune.randint(1,10000),
            "network_type": "RSNN",
            "layers": 4,
            "neuron_count": 512,
            "optimizer": "Adam",
            "lr": 0.01,
            "eps": 0.1,
            "weight_decay": 0.00001,
            "dropout": tune.grid_search([0.3]),
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.9,
            "threshold": 0.01,
            "patience": 50,
            "constantOffsetSD": tune.grid_search([0.3]),
            "randomWalkSD": 0.02,
            "whiteNoiseSD": 1.0,
            "hyperparam": hyperparam
        },


        "neuron_count_search": {
            "seed": tune.randint(1,10000),
            "neuron_count": tune.grid_search([256, 512]),
            "hyperparam": hyperparam
        },

        "inner_layer_search": {
            "seed": tune.randint(1,10000),
            "inner_layer": tune.grid_search(["fc", "conv"]),
            "hyperparam": hyperparam
        },

        "conv_layer_search": {
            "seed": tune.randint(1,10000),
            "inner_layer": "conv",
            "conv_ker_size": tune.grid_search([7, 11, 31, 51]),
            "hyperparam": hyperparam
        },

        "learning_rate_search": {
            "seed": tune.randint(1,10000),
            "lr": tune.grid_search([1e-4, 1e-3, 5e-3, 1e-2]),
            "hyperparam": hyperparam
        },

        "weight_decay_search": {
            "seed": tune.randint(1,10000),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
            "hyperparam": hyperparam
        },

        "dropout_search": {
            "seed": tune.randint(1,10000),
            "dropout": tune.grid_search([0.0, 0.2, 0.4]),
            "hyperparam": hyperparam
        },

        "time_steps_search": {
            "seed": tune.randint(1,10000),
            "train_val_timeSteps": tune.grid_search([600, 1200, 1800]),
            "hyperparam": hyperparam
        },

        "Vth_search": {
            "seed": tune.randint(1,10000),
            "Vth": tune.uniform(0.2, 0.8),
            "hyperparam": hyperparam
        },

        "noise_search": {
            "seed": tune.randint(1,10000),
            "whiteNoiseSD": tune.grid_search([0.0, 0.3, 0.6, 1.0, 1.2]),
            "constantOffsetSD": tune.grid_search([0.0, 0.3, 0.6]),
            "randomWalkSD": tune.uniform(0.0, 0.1),
            "hyperparam": hyperparam
        },

        "scheduler_search": {
            "seed": tune.randint(1,10000),
            "scheduler": tune.grid_search(["LambdaLR", "StepLR"]),
            "hyperparam": hyperparam
        },

        "scheduler_lambdaLR_search": {
            "seed": tune.randint(1,10000),
            "scheduler": "LambdaLR",
            "lr": tune.loguniform(1e-4, 1e-1),
            "step_size": tune.grid_search([10, 20, 30]),
            "gamma": tune.uniform(0.1, 0.9),
            "hyperparam": hyperparam
        },

        "surrogate_gradient_search": {
            "seed": tune.randint(1,10000),
            "surrogate_gradient": tune.grid_search(["square","multi_gaussian"]),
            "hyperparam": hyperparam
        },

        }


    return case[config_name]
