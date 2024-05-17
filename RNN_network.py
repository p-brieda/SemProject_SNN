import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, hyperparam):
        super(RNN, self).__init__()
        self.hyperparam = hyperparam
        nInputs = self.hyperparam['n_channels']
        nOutputs = self.hyperparam['n_outputs']


        # Define the RNN start state
        biDir = 2 if self.hyperparam['directionality'] == 'bidirectional' else 1
        self.rnnStartState = nn.Parameter(torch.zeros(biDir, 1, self.hyperparam['neuron_count']), requires_grad=True)
        self.initRNNState = self.rnnStartState.repeat(1, self.hyperparam['batchSize'], 1)

        # Input transformation layer (fully connected layer for each time step)
        self.inputTransform_W = nn.Parameter(torch.eye(nInputs, dtype=torch.float32), requires_grad=True)
        self.inputTransform_b = nn.Parameter(torch.zeros(nInputs, dtype=torch.float32), requires_grad=True)

        # GRU layers
        self.gru1 = nn.GRU(nInputs, self.hyperparam['neuron_count'], batch_first=True, bidirectional=(self.hyperparam['directionality'] == 'bidirectional'))
        self.gru2 = nn.GRU(self.hyperparam['neuron_count'] * biDir, self.hyperparam['neuron_count'], batch_first=True, bidirectional=(self.hyperparam['directionality'] == 'bidirectional'))

        # Linear readout layer (fully connected layer for each time step)
        self.readout_W = nn.Parameter(torch.randn(biDir * self.hyperparam['neuron_count'], nOutputs, dtype=torch.float32) * 0.05, requires_grad=True)
        self.readout_b = nn.Parameter(torch.zeros(nOutputs, dtype=torch.float32), requires_grad=True)


    def forward(self, x):
        # Transform inputs (fully connected layer for each time step)
        inputFactors = torch.matmul(x, self.inputTransform_W.unsqueeze(0).repeat(self.hyperparam['batch_size'], 1, 1)) + self.inputTransform_b

        # GRU Layer 1
        rnnOutput, _ = self.gru1(inputFactors, self.initRNNState)

        # GRU Layer 2 with skipping
        skipLen = self.hyperparam['skipLen']
        rnnOutput2, _ = self.gru2(rnnOutput[:, ::skipLen, :], self.initRNNState)

        # Linear readout layer (fully connected layer for each time step)
        logitOutput_downsample = torch.matmul(rnnOutput2, self.readout_W.unsqueeze(0).repeat(self.hyperparam['batch_size'], 1, 1)) + self.readout_b

        # Up-sample the outputs to the original time-resolution
        num_downsampled_steps = self.hyperparam['train_val_timeSteps'] // self.hyperparam['skipLen']
        expIdx = torch.arange(num_downsampled_steps, device=self.hyperparam['device'])
        expIdx = torch.repeat_interleave(expIdx, self.hyperparam['skipLen'])
        x = torch.index_select(x, 2, expIdx)

        return x