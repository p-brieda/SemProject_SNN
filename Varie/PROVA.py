

import torch
import copy
import numpy as np
import yaml
from yaml.loader import SafeLoader
import os


if torch.cuda.is_available():
    print('GPU available')

current_dir = os.getcwd()

hyperparam_file = current_dir + '\Varie\OGhyperparams.yaml'
with open(hyperparam_file) as file:
    hyperparameters = yaml.load(file, Loader=SafeLoader)

print(type(hyperparameters))
print(hyperparameters['batch_size'])