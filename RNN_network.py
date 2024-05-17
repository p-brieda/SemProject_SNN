import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, hyperparam):
        super(RNN, self).__init__()
        self.hyperparam = hyperparam
        nInputs = self.hyperparam['n_channels']
        nOutputs = self.hyperparam['n_outputs']

        # Define the RNN start state as a learnable parameter
        biDir = 2 if self.hyperparam['directionality'] == 'bidirectional' else 1
        self.biDir = biDir
        self.neuron_count = self.hyperparam['neuron_count']
        self.rnnStartState = nn.Parameter(torch.zeros(biDir, 1, self.neuron_count), requires_grad=True)
        
        # Input transformation layer (fully connected layer for each time step)
        self.inputTransform_W = nn.Parameter(torch.eye(nInputs, dtype=torch.float32), requires_grad=True)
        self.inputTransform_b = nn.Parameter(torch.zeros(nInputs, dtype=torch.float32), requires_grad=True)

        # GRU layers
        self.gru1 = nn.GRU(nInputs, self.neuron_count, batch_first=True, bidirectional=(self.hyperparam['directionality'] == 'bidirectional'))
        self.gru2 = nn.GRU(self.neuron_count * biDir, self.neuron_count, batch_first=True, bidirectional=(self.hyperparam['directionality'] == 'bidirectional'))

        # Linear readout layer (fully connected layer for each time step)
        self.readout_W = nn.Parameter(torch.randn(biDir * self.neuron_count, nOutputs, dtype=torch.float32) * 0.05, requires_grad=True)
        self.readout_b = nn.Parameter(torch.zeros(nOutputs, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.size(2)
        device = x.device

        # Transpose x to match the expected input shape for GRU (batch_size, time_steps, n_channels)
        x = x.transpose(1, 2)  # Now x is of shape (batch_size, time_steps, n_channels)

        # Transform inputs (fully connected layer for each time step)
        inputTransform_W = self.inputTransform_W.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, n_channels, n_channels]
        inputFactors = torch.matmul(x, inputTransform_W) + self.inputTransform_b  # Shape: [batch_size, time_steps, n_channels]

        # Tile the initial RNN state to match the batch size
        initRNNState = self.rnnStartState.repeat(1, batch_size, 1).to(device)

        # GRU Layer 1
        rnnOutput, hidden1 = self.gru1(inputFactors, initRNNState)

        # GRU Layer 2 with skipping
        skipLen = self.hyperparam['skipLen']
        rnnOutput2, hidden2 = self.gru2(rnnOutput[:, ::skipLen, :], initRNNState)

        # Linear readout layer (fully connected layer for each time step)
        logitOutput_downsample = torch.matmul(rnnOutput2, self.readout_W.unsqueeze(0).repeat(batch_size, 1, 1)) + self.readout_b

        # Up-sample the outputs to the original time-resolution
        num_downsampled_steps = time_steps // skipLen
        expIdx = torch.arange(num_downsampled_steps, device=device)
        expIdx = torch.repeat_interleave(expIdx, skipLen)
        logitOutput = torch.index_select(logitOutput_downsample, 1, expIdx)

        # Transpose back to the original shape (batch_size, n_outputs, time_steps)
        logitOutput = logitOutput.transpose(1, 2)

        return logitOutput