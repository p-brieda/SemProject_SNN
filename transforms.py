import numpy as np
import torch
import scipy.io
import copy

# Module containing functions for data augmentation and noise addition -- to use in the DataProcessing class

def extractSentenceSnippet(trial, nSteps, directionality='unidirectional'):
    """
    Extracts a random snippet of data from the full sentence to use for the mini-batch.
    The input 'trial' is a dictionary representing all information for a single trial.
    """
    inputs = trial['neuralData']
    targets = trial['targets']
    errWeight = trial['errWeights']
    numBinsPerTrial = trial['binsPerTrial']

    randomStart = torch.randint(low=0, high=max(numBinsPerTrial+(nSteps-100)-400, 1), size=(1,), dtype=torch.int32)
    
    inputsSnippet = inputs[randomStart:(randomStart+nSteps),:]
    targetsSnippet = targets[randomStart:(randomStart+nSteps),:]
    
    # finding the start of each character in the snippet
    charStarts = torch.where(targetsSnippet[1:,-1] - targetsSnippet[0:-1,-1]>=0.1)        
    
    def noLetters():
        ews =  torch.zeros(nSteps)
        return ews

    def atLeastOneLetter():
        firstChar = charStarts[0][0].item()
        lastChar = charStarts[0][-1].item()
        
        if directionality=='unidirectional':
            #if uni-directional, only need to blank out the first part because it's causal with a delay
            # ews --> error weights snippet
            ews =  torch.cat([torch.zeros(firstChar), 
                              errWeight[(randomStart+firstChar):(randomStart+nSteps)]], dim=0)
        else:
            #if bi-directional (acausal), we need to blank out the last incomplete character as well so that only fully complete
            #characters are included
            ews =  torch.cat([torch.zeros(firstChar), 
                              errWeight[(randomStart+firstChar):(randomStart+lastChar)],
                              torch.zeros(nSteps-lastChar)], dim=0)
            
        return ews
    # applies either noLetters or atLeastOneLetter function based on the absence or presence of characters in the snippet
    if charStarts[0].shape[0]==0:
        errWeightSnippet = noLetters()
    else:
        errWeightSnippet = atLeastOneLetter()
    
    trial['neuralData'] = inputsSnippet
    trial['targets'] = targetsSnippet
    trial['errWeights'] = errWeightSnippet
    trial['binsPerTrial'] = numBinsPerTrial




def addMeanNoise(trial, constantOffsetSD, randomWalkSD, nSteps):
    """
    Applies mean drift noise to each time step of the data in the form of constant offsets (sd=sdConstant)
    and random walk noise (sd=sdRandomWalk)
    The input 'trial' is a dictionary representing all information for a single trial.
    """  
    inputs = trial['neuralData']
    meanDriftNoise = torch.randn(1, inputs.shape[1]) * constantOffsetSD
    randomWalkNoise = torch.randn(nSteps, inputs.shape[1]) * randomWalkSD
    meanDriftNoise =  meanDriftNoise + torch.cumsum(randomWalkNoise, dim=1)
    
    trial['neuralData'] = inputs + meanDriftNoise





def addWhiteNoise(trial, whiteNoiseSD, nSteps):
    """
    Applies white noise to each time step of the data (sd=whiteNoiseSD)
    The input 'trial' is a dictionary representing all information for a single trial.
    """
    inputs = trial['neuralData']
    whiteNoise = torch.randn(nSteps, int(inputs.shape[1])) * whiteNoiseSD

    trial['neuralData'] = inputs + whiteNoise