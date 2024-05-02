

import torch
import copy
import numpy as np
import yaml
from yaml.loader import SafeLoader
import os



A = torch.rand(20, 1200, 192)
A = A.permute(0, 2, 1)
print(A.shape)

B = A[:, -1, :]
print(B.shape)


A.requires_grad = True
print(A.requires_grad)

C = A.clone()
print(C.requires_grad)

D = A.detach()
print(D.requires_grad)


c = []
for i in range(10):
    d = np.random.rand(10, 20, 30)
    c.append(d)

e = np.concatenate(c, axis=0)
print(e.shape)
