#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on May 16 2023
# @author: liaoj


from ray import tune
import numpy as np


def ray_config_dict(hyperparam,EXPERIMENT_NAME,config_name):
    case = {
        "baseline": {
            "seed": tune.randint(1,10000),
            "hyperparam": hyperparam,
        }
        }


    return case[config_name]
