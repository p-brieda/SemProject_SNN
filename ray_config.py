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
            "hyperparam": hyperparam,
        },

        "neuron_count_search": {
            "seed": tune.randint(1,10000),
            "neuron_count": tune.grid_search([256, 512, 1024]),
            "hyperparam": hyperparam
        },

        "network_search": {
            "seed": tune.randint(1,10000),
            "last_nospikes": tune.grid_search([True, False]),
            "hyperparam": hyperparam
        },

        "learning_rate_search": {
            "seed": tune.randint(1,10000),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
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
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scheduler_step_size": tune.grid_search([10, 20, 30]),
            "scheduler_gamma": tune.uniform(0.1, 0.9),
            "hyperparam": hyperparam
        },

        "surrogate_gradient_search": {
            "seed": tune.randint(1,10000),
            "surrogate_gradient": tune.grid_search(["square","multi_gaussian"]),
            "hyperparam": hyperparam
        },

        "combined_search": {
            "seed": tune.randint(1,10000),
            "scheduler": tune.grid_search(["LambdaLR", "StepLR"]),
            "learning_rate": tune.grid_search([1e-4, 5e-4,1e-3, 5e-3, 1e-2]),
            "hyperparam": hyperparam
        }


        }


    return case[config_name]
