

import torch
import copy
import numpy as np
import yaml
from yaml.loader import SafeLoader
import os



A = torch.rand(20, 1200, 192)
A = A.permute(0, 2, 1)
print(A.shape)