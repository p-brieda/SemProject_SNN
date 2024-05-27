import torch
import torch.nn as nn
import numpy as np
from layers import LIFSpike,LI_no_Spike, tdLayer, tdBatchNorm0d, tdBatchNorm0dSeq, init_surrogate_gradient
from layers import RLIFSpike, RLI_no_Spike




class Net(nn.Module):
    def __init__(self, hyperparam):
        self.hyperparam = hyperparam
        init_surrogate_gradient(hyperparam)
        super(Net, self).__init__()

        self.fc1 = tdLayer(nn.Linear(hyperparam['n_channels'], hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        if(hyperparam['batchnorm'] == 'none'):
            self.dr1 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr1 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp1 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc2 = tdLayer(nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        if(hyperparam['batchnorm'] == 'none'):
            self.dr2 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr2 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp2 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc3 = tdLayer(nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        if(hyperparam['batchnorm'] == 'none'):
            self.dr3 = tdLayer(nn.Dropout(p=hyperparam['dropout']), hyperparam = hyperparam)
        elif(hyperparam['batchnorm'] == 'tdBN'):
            self.dr3 = tdLayer(nn.Dropout(p=hyperparam['dropout']), tdBatchNorm0d(hyperparam['neuron_count'], Vth = hyperparam['Vth']), hyperparam = hyperparam)
        self.sp3 = LIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)
        
        self.fc4 = tdLayer(nn.Linear(hyperparam['neuron_count'], hyperparam['n_outputs'], bias = hyperparam['use_bias']), hyperparam = hyperparam)
        self.nospike = LI_no_Spike([hyperparam['n_outputs']], hyperparam = hyperparam)
        self.dequant = torch.quantization.DeQuantStub()



    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

 
    def forward(self, x):

        # x shape is [batch, neurons, timesteps]

        x = self.fc1(x)
        x = self.sp1(self.dr1(x))
        spikeCount1 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])

        #x = self.fc2(x)
        #x = self.sp2(self.dr2(x))
        #spikeCount2 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])

        # subsampling
        timeSteps = x.shape[2] # number of time steps of the original data
        x = self.fc3(x[:,:,0::self.hyperparam['skipLen']])
        #x = self.fc3(x)
        x = self.sp3(self.dr3(x))
        spikeCount3 = torch.mean(x, dim=(-1)).to(self.hyperparam['device'])

        x = self.fc4(x)
        x = self.nospike(x)

        #Up-sample the outputs to the original time-resolution (needed b/c layer 2 is slower).
        num_downsampled_steps = timeSteps // self.hyperparam['skipLen']
        expIdx = torch.arange(num_downsampled_steps, device=self.hyperparam['device'])
        expIdx = torch.repeat_interleave(expIdx, self.hyperparam['skipLen'])
        x = torch.index_select(x, 2, expIdx)
        

        return x,(spikeCount1, spikeCount3)
    
    def constrain(self, hyperparam):
        if(hyperparam['constrain_method']=='eval'):
            with torch.no_grad():
                self.sp1.Vth.data = torch.clamp(self.sp1.Vth,min=0)
                self.sp1.tau.data = torch.clamp(self.sp1.tau,min=0, max=1)

                self.sp2.Vth.data = torch.clamp(self.sp2.Vth,min=0)
                self.sp2.tau.data = torch.clamp(self.sp2.tau,min=0, max=1)

                self.sp3.Vth.data = torch.clamp(self.sp3.Vth,min=0)
                self.sp3.tau.data = torch.clamp(self.sp3.tau,min=0, max=1)

                self.nospike.tau.data = torch.clamp(self.nospike.tau,min=0, max=1)





class RSNNet(nn.Module):
    def __init__(self, hyperparam):
        self.hyperparam = hyperparam
        init_surrogate_gradient(hyperparam)
        super(RSNNet, self).__init__()
        #self.sp0 = LIFSpike([96],hyperparam)

        self.fc1 = nn.Linear(hyperparam['n_channels'], hyperparam['neuron_count'], bias = hyperparam['use_bias'])  # 5*5 from image dimension
        self.dr1 = nn.Dropout(p=hyperparam['dropout'])            
        if hyperparam['batchnorm'] == 'tdBN':
            self.bn1 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam)
            if(hyperparam.get('recurrent_batchnorm', False)):
                self.bnr1 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam, alpha=0.5)
                self.bn1.alpha = 0.5            
        self.sp1 = RLIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc2 = nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias'])
        self.conv2 = nn.Conv1d(1, 1, kernel_size=hyperparam['conv_ker_size'], padding=hyperparam['conv_ker_size']//2, bias = hyperparam['use_bias']) # convolutional layer
        self.frc2 = nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias'])
        self.drr2 = nn.Dropout(p=hyperparam['dropout'])
        self.dr2 = nn.Dropout(p=hyperparam['dropout'])
        if hyperparam['batchnorm'] == 'tdBN':
            self.bn2 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam)
            if(hyperparam.get('recurrent_batchnorm', False)):
                self.bnr2 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam, alpha=0.5)
                self.bn2.alpha = 0.5
        self.sp2 = RLIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)

        self.fc3 = nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias'])
        self.conv3 = nn.Conv1d(1, 1, kernel_size=hyperparam['conv_ker_size'], padding=hyperparam['conv_ker_size']//2, bias = hyperparam['use_bias'])
        self.frc3 = nn.Linear(hyperparam['neuron_count'], hyperparam['neuron_count'], bias = hyperparam['use_bias'])
        self.dr3 = nn.Dropout(p=hyperparam['dropout'])
        if(hyperparam['batchnorm'] == 'tdBN'):
            self.bn3 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam)
            if hyperparam.get('recurrent_batchnorm', False):
                self.bnr3 = tdBatchNorm0dSeq(hyperparam['neuron_count'], Vth = hyperparam['Vth'], hyperparam=hyperparam, alpha=0.5)
                self.bn3.alpha = 0.5
        self.sp3 = RLIFSpike([hyperparam['neuron_count']], hyperparam = hyperparam)
        
        self.fc4 = nn.Linear(hyperparam['neuron_count'], hyperparam['n_outputs'], bias = hyperparam['use_bias'])
        self.nospike = RLI_no_Spike([hyperparam['n_outputs']-1], hyperparam = hyperparam)
        self.dequant = torch.quantization.DeQuantStub()
        
                


    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

 
    def forward(self, x):
        steps = x.shape[-1]
        n_batches = x.shape[0]
        out = torch.zeros((n_batches,self.hyperparam['n_outputs'],steps), device=x.device)

        self.sp1.reset(batch_size = n_batches)
        self.sp2.reset(batch_size = n_batches)
        self.sp3.reset(batch_size = n_batches)
        self.nospike.reset(batch_size = n_batches)

        spikeCount1 = torch.zeros(n_batches, self.fc1.weight.shape[0], device=x.device)
        spikeCount2 = torch.zeros(n_batches, self.fc2.weight.shape[0], device=x.device)
        spikeCount3 = torch.zeros(n_batches, self.fc3.weight.shape[0], device=x.device)

        for step in range(steps):
            x_ = x[...,step]

            # FIRST LAYER
            x_ = self.fc1(x_) 
            xr = self.frc2(self.sp1.sp)
            x_ = self.dr1(x_)

            if(self.hyperparam['batchnorm'] != 'none'):
                x_ = self.bn1(x_)
                if self.hyperparam.get('recurrent_batchnorm', False):
                    xr = self.bnr1(xr)

            spikes1 = self.sp1(x_ + xr)
            spikeCount1 += spikes1

            # OPTIONAL MIDDLE LAYER
            if self.hyperparam['layers'] == 3:
                if self.hyperparam['inner_layer'] == 'fc': x_ = self.fc2(spikes1)
                elif self.hyperparam['inner_layer'] == 'conv': x_ = self.conv2(spikes1.unsqueeze(1)).squeeze(1)
                xr = self.frc2(self.sp2.sp)
                x_ = self.dr2(x_)
                if(self.hyperparam['batchnorm'] != 'none'):
                    x_ = self.bn2(x_)
                    if self.hyperparam.get('recurrent_batchnorm', False):
                        xr = self.bnr2(xr)
                
                spikes2 = self.sp2(x_ + xr)
                spikeCount2 += spikes2

            

            # SLOW LAYER
            if step % self.hyperparam['skipLen'] != 0:
                continue
            # check if we need to skip the second layer
            if self.hyperparam['layers'] == 2:
                spikes2 = spikes1
            if self.hyperparam['inner_layer'] == 'fc': x_ = self.fc3(spikes2)
            elif self.hyperparam['inner_layer'] == 'conv': x_ = self.conv3(spikes2.unsqueeze(1)).squeeze(1)
            xr = self.frc3(self.sp3.sp)
            x_ = self.dr3(x_)
            if(self.hyperparam['batchnorm'] != 'none'):
                x_ = self.bn3(x_)
            spikes3 = self.sp3(x_ + xr)
            spikeCount3 += spikes3

            # Final layer for output
            x_ = self.fc4(spikes3)
            o = x_
            if self.hyperparam['last_nospike']: 
                o[:,:-1] = self.nospike(x_[:,:-1])
            
            # expanding o with last dim = 1
            o = o.unsqueeze(-1)
            out[...,step: step + self.hyperparam['skipLen']] = o


        if self.training and self.hyperparam['batchnorm'] == 'tdBN':
            self.bn1.update()
            self.bn2.update()
            self.bn3.update()
            if self.hyperparam.get('recurrent_batchnorm', False):
                self.bnr1.update()
                self.bnr2.update()
                self.bnr3.update()

        
        spikeCount1 /= steps
        spikeCount2 /= steps
        spikeCount3 = spikeCount3 / steps * self.hyperparam['skipLen'] 

        if self.hyperparam['layers'] == 2:
            return out,(spikeCount1,spikeCount3)
        else:
            return out,(spikeCount1,spikeCount2, spikeCount3)
    
    def constrain(self, hyperparam):
        if(hyperparam['constrain_method']=='eval'):
            with torch.no_grad():
                self.sp1.Vth.data = torch.clamp(self.sp1.Vth,min=0)
                self.sp1.tau.data = torch.clamp(self.sp1.tau,min=0, max=1)

                self.sp2.Vth.data = torch.clamp(self.sp2.Vth,min=0)
                self.sp2.tau.data = torch.clamp(self.sp2.tau,min=0, max=1)

                self.sp3.Vth.data = torch.clamp(self.sp3.Vth,min=0)
                self.sp3.tau.data = torch.clamp(self.sp3.tau,min=0, max=1)

                self.nospike.tau.data = torch.clamp(self.nospike.tau,min=0, max=1)


