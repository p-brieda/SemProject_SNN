from matplotlib.pyplot import sca
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

surrogate_gradient = 'none'

def init_surrogate_gradient(hyperparam):
    global surrogate_gradient
    surrogate_gradient = hyperparam['surrogate_gradient']

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.0) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        aa = 0.5
        if(surrogate_gradient == 'square'):
            hu = abs(input) < aa
            hu = hu.float() / (2 * aa)
        elif(surrogate_gradient == 'lin'):
            hu = (-abs(input) + 1) > 0
            hu = (-abs(input) + 1) * hu.float()
        elif(surrogate_gradient == 'lin_under'):
            entire = (-abs(input) + 5) > 0
            central = (-abs(input) + 1.1) > 0
            hu = (-abs(input) + 1) * central.float() + (abs(input)/39.0 - 5.0/39.0) * (entire.float() * (1.0 - central.float()))
        elif(surrogate_gradient == 'multi_gaussian'):
            scale = 6.0
            height = .15
            lens = 0.5
            
            hu = gaussian(input, mu=0., sigma=lens) * (1. + height) \
                - gaussian(input, mu=lens, sigma=scale * lens) * height \
                - gaussian(input, mu=-lens, sigma=scale * lens) * height
            hu = hu * aa
        else:
            print('no gradient function specified')
        return grad_input * hu

spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, Vth_, tau_, reset_by_subtraction, zoned_out):
    if reset_by_subtraction:
        u_t1_n1 = tau_ * (u_t_n1 - Vth_ * o_t_n1) + W_mul_o_t1_n #next voltage = tau*(old_voltage-old outputs*Vth)+inputs (subtract old outputs)
    else:
        u_t1_n1 = tau_ * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n #next voltage = tau*old_voltage*(1-old outputs)+inputs  (reset voltage)

    u_t1_n1 = u_t1_n1 * zoned_out + (1 - zoned_out) * u_t_n1 #zoned out units are replaced by the version from the previous timestep
    
    o_t1_n1 = spikeAct(u_t1_n1 - Vth_) #next outputs
    return u_t1_n1, o_t1_n1

def state_update_no_spike(u_t_n1, o_t_n1, W_mul_o_t1_n, tau_):
    u_t1_n1 = tau_ * u_t_n1  + W_mul_o_t1_n #next voltage = tau*old_voltage+inputs
    return u_t1_n1, u_t1_n1


class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the
         last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.
    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None, hyperparam = None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.hyperparam = hyperparam
        

    def forward(self, x):
        steps = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        #variational dropout depthwise in training, but only on dropout layers
        if self.training and isinstance(self.layer,nn.Dropout) and self.hyperparam['var_dropout_s']>0:
            vardrop = torch.unsqueeze(torch.gt(torch.rand(x.shape[:-1], device=x.device),self.hyperparam['var_dropout_s']).float()/(1-self.hyperparam['var_dropout_s']), -1)
            x = x*vardrop
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

# smoothely clamps x between mi and mx
def sigmoid_clamp(x,mi, mx): 
    return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )

    
class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor 
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, inpshape = None, hyperparam = None):
        super(LIFSpike, self).__init__()
        Vth = hyperparam['Vth']
        Vth_range = hyperparam['Vth_init_range']
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.reset_by_subtraction = hyperparam['reset_by_subtraction']
        self.hyperparam = hyperparam
        self.quantize_accumulation = False
        self.quantize_membrane_pot = False
        self.scale = 1
        self.binomial = torch.distributions.binomial.Binomial(probs=1-hyperparam['var_dropout_t'])
        if(hyperparam['Vth_trainable']):
            self.Vth = nn.Parameter(torch.Tensor(*inpshape))
            nn.init.uniform_(self.Vth, Vth-Vth_range/2, Vth+Vth_range/2)
        else:
            self.Vth = torch.Tensor(*inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.Vth, Vth, Vth)

        if(hyperparam['tau_trainable']):
            # self.tau = nn.Parameter(torch.Tensor(*inpshape))
            self.tau = nn.Parameter(torch.empty(inpshape))
        else:
            # self.tau = torch.Tensor(*inpshape).to(hyperparam['device'])
            self.tau = torch.empty(inpshape).to(hyperparam['device'])

        if('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'normal'):
            nn.init.normal_(self.tau, tau, tau_range)
        elif('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'binary'):
            nn.init.uniform_(self.tau, -1 + tau, tau)
            self.tau = torch.sign(self.tau)
            self.tau = torch.heaviside(self.tau, values=nn.Parameter(torch.Tensor([0]).to(hyperparam['device'])) if hyperparam['tau_trainable'] else torch.Tensor([0]).to(hyperparam['device']))
        else:
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)
        
        #if('tau_sigmoid' in hyperparam and hyperparam['tau_sigmoid']):
        #    self.tau = sigmoid_clamp(self.tau, 0, 1)

    def quantize(self, quantize_accumulation, quantize_membrane_pot, scale, rescale):
        if(quantize_accumulation):
            self.minclamp_acc = -(2**(self.hyperparam['quantize_accumulation']-1))
            self.maxclamp_acc = 2**(self.hyperparam['quantize_accumulation']-1)-1
        self.quantize_accumulation = bool (quantize_accumulation)

        if(quantize_membrane_pot):
            self.minclamp_pot = -(2**(self.hyperparam['quantize_membrane_pot']-1))
            self.maxclamp_pot = 2**(self.hyperparam['quantize_membrane_pot']-1)-1
        self.quantize_membrane_pot = bool (quantize_membrane_pot)
        if rescale:
            self.scale = scale 
        else:
            self.scale = 1


    def forward(self, x):
        #temporal variational dropout and zoneout
        zoned_out = torch.ones(x.shape, device=x.device)
        if self.training and self.hyperparam['var_dropout_t']>0:
            if self.hyperparam['zoneout']:
                zoned_out = self.binomial.sample(x.shape).to(x.device)
            else:
                vardrop = (self.binomial.sample(x.shape[:-1]).to(x.device)/(1-self.hyperparam['var_dropout_t'])).to(x.device)

        #clamp accumulation input if quantized
        if(self.quantize_accumulation):
            x = torch.clamp(torch.round(x), min = self.minclamp_acc, max = self.maxclamp_acc)

        #noisy threshold
        if(self.training and not self.quantize_membrane_pot and self.hyperparam['noisy_threshold'] != 0):
            Vth_noise = torch.normal(0,self.hyperparam['noisy_threshold']*self.Vth).to(x.device)
        else:
            Vth_noise = torch.zeros(self.Vth.shape).to(x.device)

        steps = x.shape[-1]
        if(self.hyperparam['init_u'] == 'random'):
            u = torch.rand(x.shape[:-1] , device=x.device) * self.Vth #or maybe better to initialize high?
        elif(self.hyperparam['init_u'] == 'zero'):
            u = torch.zeros(x.shape[:-1] , device=x.device)
        elif(self.hyperparam['init_u'] == 'Vth'):
            u = torch.ones(x.shape[:-1] , device=x.device) * (self.Vth - 1e-6)
        out = torch.zeros(x.shape, device=x.device)

        #constrain tau and Vth
        if(self.hyperparam['constrain_method']=='forward'):
            Vth = torch.clamp(self.Vth, min = 0.0)
            tau = torch.clamp(self.tau, min = 0.0, max = 1.0)
        elif(self.hyperparam['constrain_method']=='always'):
            self.Vth.data = torch.clamp(self.Vth, min = 0.0)
            Vth = self.Vth
            self.tau.data = torch.clamp(self.tau, min = 0.0, max = 1.0)
            tau = self.tau
        elif(self.hyperparam['constrain_method']=='none' or self.hyperparam['constrain_method']=='eval'):
            Vth = self.Vth
            tau = self.tau

        if self.hyperparam.get('tau_sigmoid', False):
            sigmoid = torch.nn.Sigmoid()
            tau = sigmoid(tau)
            
        for step in range(steps): 
            #max(step-1,0) since step is 0 in the first iteration and we need to give the spikes from the last timestep, we give zero instead.
            u, out[...,step] = state_update(u, out[..., max(step-1, 0)], x[..., step], (Vth+Vth_noise)*self.scale, tau, self.reset_by_subtraction, zoned_out[...,step])
            if(self.quantize_membrane_pot):
                u = torch.clamp(torch.round(u), min = self.minclamp_pot, max = self.maxclamp_pot)
            if self.training and self.hyperparam['var_dropout_t']>0 and not self.hyperparam['zoneout']:
                u = u*vardrop
        return out


#Persistent Spiking Neuron
class PLIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor 
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, inpshape = None, hyperparam = None):
        super(PLIFSpike, self).__init__()
        Vth = hyperparam['Vth']
        Vth_range = hyperparam['Vth_init_range']
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.reset_by_subtraction = hyperparam['reset_by_subtraction']
        self.hyperparam = hyperparam
        self.quantize_accumulation = False
        self.quantize_membrane_pot = False
        self.scale = 1
        self.binomial = torch.distributions.binomial.Binomial(probs=1-hyperparam['var_dropout_t'])
        if(hyperparam['Vth_trainable']):
            self.Vth = nn.Parameter(torch.Tensor(*inpshape))
            nn.init.uniform_(self.Vth, Vth-Vth_range/2, Vth+Vth_range/2)
        else:
            self.Vth = torch.Tensor(*inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.Vth, Vth, Vth)

        if(hyperparam['tau_trainable']):
            self.tau = nn.Parameter(torch.Tensor(*inpshape))
        else:
            self.tau = torch.Tensor(*inpshape).to(hyperparam['device'])

        if('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'normal'):
            nn.init.normal_(self.tau, tau, tau_range)
        elif('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'binary'):
            nn.init.uniform_(self.tau, -1 + tau, tau)
            self.tau = torch.sign(self.tau)
            self.tau = torch.heaviside(self.tau, values=nn.Parameter(torch.Tensor([0]).to(hyperparam['device'])) if hyperparam['tau_trainable'] else torch.Tensor([0]).to(hyperparam['device']))
        else:
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)

        dim = tuple(np.concatenate(([hyperparam['batch_size']], self.inpshape)))
        if(self.hyperparam['init_u'] == 'random'):
            self.u = torch.rand(dim , device=self.hyperparam['device']) * self.Vth #or maybe better to initialize high?
        elif(self.hyperparam['init_u'] == 'zero'):
            self.u = torch.zeros(dim , device=self.hyperparam['device'])
        elif(self.hyperparam['init_u'] == 'Vth'):
            self.u = torch.ones(dim, device=self.hyperparam['device']) * (self.Vth - 1e-6)
            
        self.sp = torch.zeros(dim, device=self.hyperparam['device'])
        
        #if('tau_sigmoid' in hyperparam and hyperparam['tau_sigmoid']):
        #    self.tau = sigmoid_clamp(self.tau, 0, 1)

    def quantize(self, quantize_accumulation, quantize_membrane_pot, scale, rescale):
        if(quantize_accumulation):
            self.minclamp_acc = -(2**(self.hyperparam['quantize_accumulation']-1))
            self.maxclamp_acc = 2**(self.hyperparam['quantize_accumulation']-1)-1
        self.quantize_accumulation = bool (quantize_accumulation)

        if(quantize_membrane_pot):
            self.minclamp_pot = -(2**(self.hyperparam['quantize_membrane_pot']-1))
            self.maxclamp_pot = 2**(self.hyperparam['quantize_membrane_pot']-1)-1
        self.quantize_membrane_pot = bool (quantize_membrane_pot)
        if rescale:
            self.scale = scale 
        else:
            self.scale = 1


    def forward(self, x):
        #temporal variational dropout and zoneout
        zoned_out = torch.ones(x.shape, device=x.device)
        if self.training and self.hyperparam['var_dropout_t']>0:
            if self.hyperparam['zoneout']:
                zoned_out = self.binomial.sample(x.shape).to(x.device)
            else:
                vardrop = (self.binomial.sample(x.shape[:-1]).to(x.device)/(1-self.hyperparam['var_dropout_t'])).to(x.device)

        #clamp accumulation input if quantized
        if(self.quantize_accumulation):
            x = torch.clamp(torch.round(x), min = self.minclamp_acc, max = self.maxclamp_acc)

        #noisy threshold
        if(self.training and not self.quantize_membrane_pot and self.hyperparam['noisy_threshold'] != 0):
            Vth_noise = torch.normal(0,self.hyperparam['noisy_threshold']*self.Vth).to(x.device)
        else:
            Vth_noise = torch.zeros(self.Vth.shape).to(x.device)

        steps = x.shape[-1]
        if self.hyperparam.get('keep_membrane_potential', False):
            u = self.u[:x.shape[0],:] #trunkate to batch size
            sp = self.sp[:x.shape[0],:] #trunkate to batch size
            out = torch.zeros(x.shape, device=x.device)
        else:           
            if(self.hyperparam['init_u'] == 'random'):
                u = torch.rand(x.shape[:-1] , device=x.device) * self.Vth #or maybe better to initialize high?
            elif(self.hyperparam['init_u'] == 'zero'):
                u = torch.zeros(x.shape[:-1] , device=x.device)
            elif(self.hyperparam['init_u'] == 'Vth'):
                u = torch.ones(x.shape[:-1] , device=x.device) * (self.Vth - 1e-6)
            out = torch.zeros(x.shape, device=x.device)

        #constrain tau and Vth
        if(self.hyperparam['constrain_method']=='forward'):
            Vth = torch.clamp(self.Vth, min = 0.0)
            tau = torch.clamp(self.tau, min = 0.0, max = 1.0)
        elif(self.hyperparam['constrain_method']=='always'):
            self.Vth.data = torch.clamp(self.Vth, min = 0.0)
            Vth = self.Vth
            self.tau.data = torch.clamp(self.tau, min = 0.0, max = 1.0)
            tau = self.tau
        elif(self.hyperparam['constrain_method']=='none' or self.hyperparam['constrain_method']=='eval'):
            Vth = self.Vth
            tau = self.tau

        if self.hyperparam.get('tau_sigmoid', False):
            sigmoid = torch.nn.Sigmoid()
            tau = sigmoid(tau)

        for step in range(steps): 
            #for first timestep give spikes from last batch
            if step == 0: 
                u, out[...,step] = state_update(u, sp, x[..., step], (Vth+Vth_noise)*self.scale, tau, self.reset_by_subtraction, zoned_out[...,step])
            else:
                u, out[...,step] = state_update(u, out[..., step], x[..., step], (Vth+Vth_noise)*self.scale, tau, self.reset_by_subtraction, zoned_out[...,step])
            if(self.quantize_membrane_pot):
                u = torch.clamp(torch.round(u), min = self.minclamp_pot, max = self.maxclamp_pot)
            if self.training and self.hyperparam['var_dropout_t']>0 and not self.hyperparam['zoneout']:
                u = u*vardrop
        
        self.u = u.detach()
        self.sp = out[..., -1].detach()
        return out

class RLIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor 
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, inpshape = None, hyperparam = None):
        super(RLIFSpike, self).__init__()
        Vth = hyperparam['Vth']
        Vth_range = hyperparam['Vth_init_range']
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.reset_by_subtraction = hyperparam['reset_by_subtraction']
        self.hyperparam = hyperparam
        self.quantize_accumulation = False
        self.quantize_membrane_pot = False
        self.scale = 1
        self.binomial = torch.distributions.binomial.Binomial(probs=1-hyperparam['var_dropout_t'])
        if(hyperparam['Vth_trainable']):
            self.Vth = nn.Parameter(torch.Tensor(*inpshape))
            nn.init.uniform_(self.Vth, Vth-Vth_range/2, Vth+Vth_range/2)
        else:
            self.Vth = torch.Tensor(*inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.Vth, Vth, Vth)

        if(hyperparam['tau_trainable']):
            self.tau = nn.Parameter(torch.Tensor(*inpshape))
        else:
            self.tau = torch.Tensor(*inpshape).to(hyperparam['device'])

        if('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'normal'):
            nn.init.normal_(self.tau, tau, tau_range)
        elif('tau_init_distribution' in hyperparam and hyperparam['tau_init_distribution'] == 'binary'):
            nn.init.uniform_(self.tau, -1 + tau, tau)
            self.tau = torch.sign(self.tau)
            self.tau = torch.heaviside(self.tau, values=nn.Parameter(torch.Tensor([0]).to(hyperparam['device'])) if hyperparam['tau_trainable'] else torch.Tensor([0]).to(hyperparam['device']))
        else:
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)
        
        self.reset(batch_size=hyperparam['batch_size'])
        
        #if('tau_sigmoid' in hyperparam and hyperparam['tau_sigmoid']):
        #    self.tau = sigmoid_clamp(self.tau, 0, 1)        
        
        
    def reset(self, batch_size = 128):
        dim = tuple(np.concatenate(([batch_size], self.inpshape)))
        if(self.hyperparam['init_u'] == 'random'):
            self.u = torch.rand(dim , device=self.hyperparam['device']) * self.Vth #or maybe better to initialize high?
        elif(self.hyperparam['init_u'] == 'zero'):
            self.u = torch.zeros(dim , device=self.hyperparam['device'])
        elif(self.hyperparam['init_u'] == 'Vth'):
            self.u = torch.ones(dim, device=self.hyperparam['device']) * (self.Vth - 1e-6)
            
        self.sp = torch.zeros(dim, device=self.hyperparam['device'])

    def quantize(self, quantize_accumulation, quantize_membrane_pot, scale, rescale):
        if(quantize_accumulation):
            self.minclamp_acc = -(2**(self.hyperparam['quantize_accumulation']-1))
            self.maxclamp_acc = 2**(self.hyperparam['quantize_accumulation']-1)-1
        self.quantize_accumulation = bool (quantize_accumulation)

        if(quantize_membrane_pot):
            self.minclamp_pot = -(2**(self.hyperparam['quantize_membrane_pot']-1))
            self.maxclamp_pot = 2**(self.hyperparam['quantize_membrane_pot']-1)-1
        self.quantize_membrane_pot = bool (quantize_membrane_pot)
        if rescale:
            self.scale = scale 
        else:
            self.scale = 1
            
    


    def forward(self, x):
        #temporal variational dropout and zoneout
        zoned_out = torch.ones(x.shape, device=x.device)
        if self.training and self.hyperparam['var_dropout_t']>0:
            if self.hyperparam['zoneout']:
                zoned_out = self.binomial.sample(x.shape).to(x.device)
            else:
                vardrop = (self.binomial.sample(x.shape).to(x.device)/(1-self.hyperparam['var_dropout_t'])).to(x.device)

        #clamp accumulation input if quantized
        if(self.quantize_accumulation):
            x = torch.clamp(torch.round(x), min = self.minclamp_acc, max = self.maxclamp_acc)

        #noisy threshold
        if(self.training and not self.quantize_membrane_pot and self.hyperparam['noisy_threshold'] != 0):
            Vth_noise = torch.normal(0,self.hyperparam['noisy_threshold']*self.Vth).to(x.device)
        else:
            Vth_noise = torch.zeros(self.Vth.shape).to(x.device)

        out = torch.zeros(x.shape, device=x.device)

        #constrain tau and Vth
        if(self.hyperparam['constrain_method']=='forward'):
            Vth = torch.clamp(self.Vth, min = 0.0)
            tau = torch.clamp(self.tau, min = 0.0, max = 1.0)
        elif(self.hyperparam['constrain_method']=='always'):
            self.Vth.data = torch.clamp(self.Vth, min = 0.0)
            Vth = self.Vth
            self.tau.data = torch.clamp(self.tau, min = 0.0, max = 1.0)
            tau = self.tau
        elif(self.hyperparam['constrain_method']=='none' or self.hyperparam['constrain_method']=='eval'):
            Vth = self.Vth
            tau = self.tau

        if self.hyperparam.get('tau_sigmoid', False):
            sigmoid = torch.nn.Sigmoid()
            tau = sigmoid(tau)
            
        self.u, self.sp = state_update(self.u, self.sp, x, (Vth+Vth_noise)*self.scale, tau, self.reset_by_subtraction, zoned_out)
        self.sp = torch.squeeze(self.sp, 0) #get rid of extra dimension

        if(self.quantize_membrane_pot):
            u = torch.clamp(torch.round(u), min = self.minclamp_pot, max = self.maxclamp_pot)
        if self.training and self.hyperparam['var_dropout_t']>0 and not self.hyperparam['zoneout']:
            u = u*vardrop
        return self.sp


class LI_no_Spike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates . It can be considered as an activation function and is used similar to ReLU. The input tensor 
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self, inpshape = None, hyperparam = None):
        super(LI_no_Spike, self).__init__()
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.hyperparam = hyperparam
        self.quantize_accumulation = False
        self.quantize_membrane_pot = False
        self.Vth = None
        self.scale = 1
        if(hyperparam['tau_trainable']):
            # self.tau = nn.Parameter(torch.Tensor(inpshape))
            # self.tau = nn.Parameter(torch.Tensor(*inpshape))
            self.tau = nn.Parameter(torch.empty(inpshape))
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)
        else:
            # self.tau = torch.Tensor(inpshape).to(hyperparam['device'])
            # self.tau = torch.Tensor(*inpshape).to(hyperparam['device'])
            self.tau = torch.empty(inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)  

    def quantize(self, quantize_accumulation, quantize_membrane_pot, scale, rescale):
        if(quantize_accumulation):
            self.minclamp_acc = -(2**(self.hyperparam['quantize_accumulation']-1))
            self.maxclamp_acc = 2**(self.hyperparam['quantize_accumulation']-1)-1
        self.quantize_accumulation = bool (quantize_accumulation)

        if(quantize_membrane_pot):
            self.minclamp_pot = -(2**(self.hyperparam['quantize_membrane_pot']-1))
            self.maxclamp_pot = 2**(self.hyperparam['quantize_membrane_pot']-1)-1
        self.quantize_membrane_pot = bool (quantize_membrane_pot)
        if(rescale):
            self.scale = scale
        else:
            self.scale = 1

    def forward(self, x):
        steps = x.shape[-1]
        u = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)

        #clamp accumulation input if quantized
        if(self.quantize_accumulation):
            x = torch.clamp(torch.round(x), min = self.minclamp_acc, max = self.maxclamp_acc)


        #constrain tau and Vth
        if(self.hyperparam['constrain_method']=='forward'):
            tau = torch.clamp(self.tau, min = 0.0, max = 1.0)
        elif(self.hyperparam['constrain_method']=='always'):
            self.tau.data = torch.clamp(self.tau, min = 0.0, max = 1.0)
            tau = self.tau  
        elif(self.hyperparam['constrain_method']=='none' or self.hyperparam['constrain_method']=='eval'):
            tau = self.tau
        for step in range(steps):
            u, out[..., step] = state_update_no_spike(u, out[..., max(step-1, 0)], x[..., step], tau)
            if(self.quantize_membrane_pot):
                u = torch.clamp(torch.round(u), min = self.minclamp_pot, max = self.maxclamp_pot)
        return out/self.scale


class RLI_no_Spike(nn.Module):
    def __init__(self, inpshape = None, hyperparam = None):
        super(RLI_no_Spike, self).__init__()
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.hyperparam = hyperparam
        self.quantize_accumulation = False
        self.quantize_membrane_pot = False
        self.Vth = None
        self.scale = 1
        if(hyperparam['tau_trainable']):
            self.tau = nn.Parameter(torch.Tensor(inpshape))
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)
        else:
            self.tau = torch.Tensor(inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau-tau_range/2, tau+tau_range/2)  

        self.reset(batch_size = hyperparam['batch_size'])

            
    def reset(self, batch_size=128):
        dim = tuple(np.concatenate(([batch_size], self.inpshape)))
        self.u = torch.zeros(dim , device=self.hyperparam['device'])
        self.out = torch.zeros(dim, device=self.hyperparam['device'])

    def quantize(self, quantize_accumulation, quantize_membrane_pot, scale, rescale):
        if(quantize_accumulation):
            self.minclamp_acc = -(2**(self.hyperparam['quantize_accumulation']-1))
            self.maxclamp_acc = 2**(self.hyperparam['quantize_accumulation']-1)-1
        self.quantize_accumulation = bool (quantize_accumulation)

        if(quantize_membrane_pot):
            self.minclamp_pot = -(2**(self.hyperparam['quantize_membrane_pot']-1))
            self.maxclamp_pot = 2**(self.hyperparam['quantize_membrane_pot']-1)-1
        self.quantize_membrane_pot = bool (quantize_membrane_pot)
        if(rescale):
            self.scale = scale
        else:
            self.scale = 1

    def forward(self, x):
        steps = x.shape[-1]
        
        #clamp accumulation input if quantized
        if(self.quantize_accumulation):
            x = torch.clamp(torch.round(x), min = self.minclamp_acc, max = self.maxclamp_acc)


        #constrain tau and Vth
        if(self.hyperparam['constrain_method']=='forward'):
            tau = torch.clamp(self.tau, min = 0.0, max = 1.0)
        elif(self.hyperparam['constrain_method']=='always'):
            self.tau.data = torch.clamp(self.tau, min = 0.0, max = 1.0)
            tau = self.tau  
        elif(self.hyperparam['constrain_method']=='none' or self.hyperparam['constrain_method']=='eval'):
            tau = self.tau
            
            
        self.u, self.out = state_update_no_spike(self.u, self.out, x, tau)
        if(self.quantize_membrane_pot):
            self.u = torch.clamp(torch.round(self.u), min = self.minclamp_pot, max = self.maxclamp_pot)
        return self.out/self.scale



class tdBatchNorm2d(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well 
        when doing BN. Expects inputs and outputs as (Batch, Channel, Height, Width, Time)
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth = 0.0):
        super(tdBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False

    def forward(self, input):
        if(self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input
class tdBatchNorm1d(nn.BatchNorm1d):
    """1d version of tdBatchNorm2d: Only difference is the input and output shapes: Expects inputs and outputs as (Batch, Channel, Width, Time)
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth = 0.0):
        super(tdBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False

    def forward(self, input):
        if(self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
class tdBatchNorm0d(nn.BatchNorm1d):
    """0d version of tdBatchNorm2d: Only difference is the input and output shapes: Expects inputs and outputs as (Batch, Channel, Time)
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth = 0.0):
        super(tdBatchNorm0d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False

    def forward(self, input):
        if(self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2])
            # use biased var in train
            var = input.var([0, 2], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None] + self.bias[None, :, None]

        return input



class tdBatchNorm0dSeq(nn.BatchNorm1d):
    """0d version of tdBatchNorm2d: Only difference is the input and output shapes: Expects inputs and outputs as (Batch, Channel, Time)
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth = 0.0, hyperparam = None, use_batch_stats = False):
        super(tdBatchNorm0dSeq, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False
        self.step_memory = torch.zeros((hyperparam['batch_size'], num_features, hyperparam['steps']), device = hyperparam['device'])
        self.current_step = 0
        self.batch_size = hyperparam['batch_size']
        self.use_batch_stats = use_batch_stats

    def forward(self, input):
        if self.training and not self.deactivated:
            # for abnormal batch sizes (when a batch cant be filled completely at end of epoch)
            if self.current_step == 0 and input.shape[0] != self.step_memory.shape[0]:
                self.step_memory = torch.zeros((input.shape[0], self.step_memory.shape[1], self.step_memory.shape[2]), device = self.step_memory.device)

            self.step_memory[..., self.current_step] = input
            self.current_step += 1

        if(self.deactivated or not hasattr(self, 'running_mean')): #don't do BN before first update
            return input

        if self.use_batch_stats and self.training:
            with torch.no_grad():
                mean = self.step_memory[...,:self.current_step].mean([0, 2])
                var = self.step_memory[...,:self.current_step].var([0, 2], unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input

    def update(self):
        self.current_step = 0

        if not self.training:
            print("Warning: BatchNorm update in eval mode")
            return

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            with torch.no_grad():
                mean = self.step_memory.mean([0, 2])
                # use biased var in train
                var = self.step_memory.var([0, 2], unbiased=False)
                n = self.step_memory.numel() / self.step_memory.size(1)
            
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var

            

