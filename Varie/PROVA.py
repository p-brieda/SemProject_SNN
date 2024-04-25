

import torch
import copy
import numpy as np


A = {'primo':torch.ones(10), 'secondo':torch.zeros(10)}
b = {key:data[5] for key, data in A.items()}
b['primo'] = b['primo'] + 10

C = np.arange(50)

D = np.array([1, 2, 3, 4])
e = [1, 2, 3]

prob = np.array([0.1, 0.2, 0.3, 0.2])
prob_norm = prob / prob.sum()


print(np.arange(10))


E = torch.rand(10,10)
F = torch.rand(10,10)
#print(E)
f = torch.where(E[:,-1] - F[:,-1]>0)
#print(type(f))
#print(f)
#print(f[0][0].item())
#print(f[0][-1].item())
#print(f[0].shape[0])



class ciao():
    def __init__(self, param1, param2):
        self.a = param1
        self.b = param2
        self.list1 = []
        self.list2 = []
        self.listAll = [self.list1, self.list2]
    
    def get(self):
        return [self.a, self.b]
    

    def app(self):
        for i in range(10):
            self.list1.append(i*10)
            self.list2.append(i*20)
        print(self.listAll[0], self.listAll[1])
        
    

c = [ciao(param1=10, param2=20), ciao(param1=30, param2=40)]
d = c[1].get()

e = ciao(param1=10, param2=20)
e.app()


import ray
from ray import tune