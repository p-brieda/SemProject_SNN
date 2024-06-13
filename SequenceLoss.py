import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss function
# The channels dimension and the temporal dimension are (B x C x T)
# L2 regularization is not included in the loss class since it will be added in the optimizer

class SequenceLoss(nn.Module):
    def __init__(self, hyperparam):
        super().__init__()
        self.hyperparams = hyperparam

    def forward(self, logitOutput, batchTargets, batchWeight):
        # Handling the output delay
        labels = batchTargets[:, :, 0:-self.hyperparams['outputDelay']]
        logits = logitOutput[:, :, self.hyperparams['outputDelay']:]
        bw = batchWeight[:, 0:-self.hyperparams['outputDelay']]

        transOut = logits[:,-1,:]
        transLabel = labels[:,-1,:]
        logits = logits[:,0:-1,:]
        labels = labels[:,0:-1,:]

        # Cross-entropy character probability loss
        ceLoss = F.cross_entropy(logits, labels, reduction='none')
        totalErr = torch.mean(torch.sum(bw * ceLoss, dim=1) / self.hyperparams['train_val_timeSteps'])

        # Character start signal loss
        sqErrLoss = torch.square(torch.sigmoid(transOut)-transLabel)
        totalErr += 5*torch.mean(torch.sum(sqErrLoss,dim=1)/self.hyperparams['train_val_timeSteps'])

        return totalErr
