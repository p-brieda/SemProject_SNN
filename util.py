import numpy as np
import torch
import scipy.io
import copy
import os
from datetime import datetime




def prepareDataCubesForRNN(sentenceFile, singleLetterFile, labelFile, cvPartitionFile, sessionName, rnnBinSize, nTimeSteps, isTraining):
    """
    Loads raw data & HMM labels and returns training and validation data cubes for RNN training (or inference). 
    Normalizes the neural activity using the single letter means & standard deviations.
    Does some additional pre-processing, including zero-padding the data and cutting off the end of the last character if it is too long.
    (Long pauses occur at the end of some sentences since T5 often paused briefly after finishing instead of 
    continuing immediately to the next sentence).
    """
    sentenceDat = scipy.io.loadmat(sentenceFile)
    slDat = scipy.io.loadmat(singleLetterFile)
    labelsDat = scipy.io.loadmat(labelFile)
    cvPart = scipy.io.loadmat(cvPartitionFile)
                      
    errWeights = 1-labelsDat['ignoreErrorHere']
    charProbTarget = labelsDat['charProbTarget']
    charStartTarget = labelsDat['charStartTarget'][:,:,np.newaxis]

    #Here we update the error weights to ignore time bins outside of the sentence
    for t in range(labelsDat['timeBinsPerSentence'].shape[0]):
        errWeights[t,labelsDat['timeBinsPerSentence'][t,0]:] = 0

        # Also, we cut off the end of the trial if there is a very long pause after the last letter - this could hurt during training
        # Max duration at the end will be the time at which the last character  started + 150 milliseconds
        maxPause = 150
        lastCharStart = np.argwhere(charStartTarget[t,:]>0.5)
        errWeights[t,(lastCharStart[-1,0]+maxPause):] = 0
        labelsDat['timeBinsPerSentence'][t,0] = (lastCharStart[-1,0]+maxPause)

    # The two targets (probability and character start).
    #The rest of the code then assumes that the last column is the character start target.
    combinedTargets = np.concatenate([charProbTarget, charStartTarget], axis=2)

    nRNNOutputs = combinedTargets.shape[2] 
    binsPerTrial = np.round(labelsDat['timeBinsPerSentence']/rnnBinSize).astype(np.int32)
    binsPerTrial = np.squeeze(binsPerTrial)

    #get normalized neural data cube for the sentences
    neuralData = normalizeSentenceDataCube(sentenceDat, slDat)

    #bin the data across the time axis
    if rnnBinSize>1:
        neuralData = binTensor(neuralData, rnnBinSize)
        combinedTargets = binTensor(combinedTargets, rnnBinSize)
        errWeights = np.squeeze(binTensor(errWeights[:,:,np.newaxis], rnnBinSize))

    #zero padding
    if isTraining:
        #train mode, add some extra zeros to the end so that we can begin snippets near the end of sentences
        edgeSpace = (nTimeSteps-100)
        padTo = neuralData.shape[1]+edgeSpace*2
        
        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:,edgeSpace:(edgeSpace+neuralData.shape[1]),:] = neuralData
        padCombinedTargets[:,edgeSpace:(edgeSpace+combinedTargets.shape[1]),:] = combinedTargets
        padErrWeights[:,edgeSpace:(edgeSpace+errWeights.shape[1])] = errWeights
    else:
        #inference mode, pad up to the specified time steps (which should be > than the data cube length, and a multiple of skipLen)
        padTo = nTimeSteps

        padNeuralData = np.zeros([neuralData.shape[0], padTo, neuralData.shape[2]])
        padCombinedTargets = np.zeros([combinedTargets.shape[0], padTo, combinedTargets.shape[2]])
        padErrWeights = np.zeros([errWeights.shape[0], padTo])

        padNeuralData[:,0:neuralData.shape[1],:] = neuralData
        padCombinedTargets[:,0:combinedTargets.shape[1],:] = combinedTargets
        padErrWeights[:,0:errWeights.shape[1]] = errWeights

    #gather the train/validation fold indices
    cvIdx = {}                          
    cvIdx['trainIdx'] = np.squeeze(cvPart[sessionName+'_train'])
    cvIdx['testIdx'] = np.squeeze(cvPart[sessionName+'_test'])

    padNeuralData = torch.from_numpy(padNeuralData).type(torch.FloatTensor)
    padCombinedTargets = torch.from_numpy(padCombinedTargets).type(torch.FloatTensor)
    padErrWeights = torch.from_numpy(padErrWeights).type(torch.FloatTensor)
    binsPerTrial = torch.from_numpy(binsPerTrial).type(torch.IntTensor)

    return padNeuralData, padCombinedTargets, padErrWeights, binsPerTrial, cvIdx




def unfoldDataCube(trials_train, trials_val, neuralData, targets, errWeights, binsPerTrial, cvIdx, dayIdx):
    # Function that takes the data cubes and unfolds them

    for tIdx in cvIdx['trainIdx']:

        trials_train['neuralData'].append(neuralData[tIdx,:,:])
        trials_train['targets'].append(targets[tIdx,:,:])
        trials_train['errWeights'].append(errWeights[tIdx,:])
        trials_train['binsPerTrial'].append(binsPerTrial[tIdx])
        trials_train['dayIdx'].append(dayIdx)

    
    for vIdx in cvIdx['testIdx']:
        
        trials_val['neuralData'].append(neuralData[vIdx,:,:])
        trials_val['targets'].append(targets[vIdx,:,:])
        trials_val['errWeights'].append(errWeights[vIdx,:])
        trials_val['binsPerTrial'].append(binsPerTrial[vIdx])
        trials_val['dayIdx'].append(dayIdx)
    



def normalizeSentenceDataCube(sentenceDat, singleLetterDat):
    """
    Normalizes the neural data cube by subtracting means and dividing by the standard deviation. 
    Important: we use means and standard deviations from the single letter data. This is needed since we 
    initialize the HMM parameters using the single letter data, so the sentence data needs to be normalized in the same way. 
    """
    neuralCube = sentenceDat['neuralActivityCube'].astype(np.float64)

    #subtract block-specific means from each trial to counteract the slow drift in feature means over time
    for b in range(sentenceDat['blockList'].shape[0]):
        trialsFromThisBlock = np.squeeze(sentenceDat['sentenceBlockNums']==sentenceDat['blockList'][b])
        trialsFromThisBlock = np.argwhere(trialsFromThisBlock)

        closestIdx = np.argmin(np.abs(singleLetterDat['blockList'].astype(np.int32) - sentenceDat['blockList'][b].astype(np.int32)))
        blockMeans = singleLetterDat['meansPerBlock'][closestIdx,:]

        neuralCube[trialsFromThisBlock,:,:] -= blockMeans[np.newaxis,np.newaxis,:]

    #divide by standard deviation to normalize the units
    neuralCube = neuralCube / singleLetterDat['stdAcrossAllData'][np.newaxis,:,:]
    
    return neuralCube




def binTensor(data, binSize):
    """
    A simple utility function to bin a 3d numpy tensor along axis 1 (the time axis here). Data is binned by
    taking the mean across a window of time steps. 
    
    Args:
        data (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        binSize (int): The bin size in # of time steps
        
    Returns:
        binnedTensor (tensor : B x S x N): A 3d tensor with batch size B, time bins S, and number of features N.
                                           S = floor(T/binSize)
    """
    
    nBins = np.floor(data.shape[1]/binSize).astype(int)
    
    sh = np.array(data.shape)
    sh[1] = nBins
    binnedTensor = np.zeros(sh)
    
    binIdx = np.arange(0,binSize).astype(int)
    for t in range(nBins):
        binnedTensor[:,t,:] = np.mean(data[:,binIdx,:],axis=1)
        binIdx += binSize;
    
    return binnedTensor



# function for the creation of the arguments dictionary
def getDefaultArgs():
    args = {}

    args['gpuNumber'] = '0'

    rootDir = 'C:/Users/pietr/OneDrive/Documenti/PIETRO/ETH/SS24/Semester_project/handwritingBCIData/'

    dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']
    
    cvPart = 'HeldOutBlocks'

    for x in range(len(dataDirs)):
        args['sentencesFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat'
        args['singleLettersFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/singleLetters.mat'
        args['labelsFile_'+str(x)] = rootDir+'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
        args['syntheticDatasetDir_'+str(x)] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
        args['cvPartitionFile_'+str(x)] = rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat'
        args['sessionName_'+str(x)] = dataDirs[x]

    #Specifies how many 10 ms time steps to combine a single bin for RNN processing                              
    args['rnnBinSize'] = 2

    #Applies Gaussian smoothing if equal to 1                             
    args['smoothInputs'] = 1

    #How many bins to delay the output. Some delay is needed in order to give the RNN enough time to see the entire character
    #before deciding on its identity. Default is 1 second (50 bins).
    args['outputDelay'] = 50

    #Can be 'unidrectional' (causal) or 'bidirectional' (acausal)                              
    args['directionality'] = 'unidirectional'

    #standard deivation of the constant-offset firing rate drift noise                             
    args['constantOffsetSD'] = 0.6

    #standard deviation of the random walk firing rate drift noise                             
    args['randomWalkSD'] = 0.02

    #standard deivation of the white noise added to the inputs during training                            
    args['whiteNoiseSD'] = 1.2

    #this seed is set for numpy and tensorflow when the class is initialized                             
    args['seed'] = datetime.now().microsecond

    #number of time steps to use in the minibatch (1200 = 24 seconds) in training/validation mode                        
    args['train_val_timeSteps'] = 1200

    #number of time steps to use in the minibatch (1200 = 24 seconds) in test mode
    args['test_timeSteps'] = 7500

    #number of sentence snippets to include in the minibatch                             
    args['batchSize'] = 64

    print("Please set:", "/n","args['outputDir']")

    return args











