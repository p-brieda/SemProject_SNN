import numpy as np
import torch
import scipy.io
import copy
import os
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
from Evaluation import evaluateSNNOutput, wer, decodeCharStr
import logging
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import torch.nn as nn


# function for the creation of the arguments dictionary
def getDefaultHyperparams():

    current_dir = os.getcwd()
    hyperparam_file = current_dir + '/hyperparams.yaml'
    with open(hyperparam_file) as file:
        hyperparams = yaml.load(file, Loader=SafeLoader)
    
    
    rootDir = 'C:/Users/pietr/OneDrive/Documenti/PIETRO/ETH/SS24/Semester_project/handwritingBCIData/'
    hyperparams['system'] = 'Windows'
    hyperparams['prepared_data_dir'] = 'C:/Users/pietr/OneDrive/Documenti/PIETRO/ETH/SS24/Semester_project/SNN_project/dataset/'

    # Handle different OS (Linux and Windows)
    if not os.path.exists(rootDir): 
        rootDir = os.path.expanduser('~') + '/Semester_project/handwritingBCIData/'
        hyperparams['system'] = 'Linux'
        hyperparams['output_report'] = '/home/sem24f8/Semester_project/SNN_Project/SemProject_SNN/Report.txt'
        hyperparams['results_dir'] = '/home/sem24f8/Semester_project/SNN_Project/SemProject_SNN/trainOutputs/'
        hyperparams['output_plots'] = '/home/sem24f8/Semester_project/SNN_Project/SemProject_SNN/Plots/'
        hyperparams['output_csv'] = '/home/sem24f8/Semester_project/SNN_Project/SemProject_SNN/CSV/'
        hyperparams['save_model_dir'] = '/home/sem24f8/Semester_project/SNN_Project/SemProject_SNN/Model/'
        hyperparams['prepared_data_dir'] = '/scratch/sem24f8/dataset/'


    
    # Define the data directories
    dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']
    
    hyperparams['dataDirs'] = dataDirs
    
    cvPart = 'HeldOutBlocks'

    for x in range(len(dataDirs)):
        hyperparams['sentencesFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat'
        hyperparams['singleLettersFile_'+str(x)] = rootDir+'Datasets/'+dataDirs[x]+'/singleLetters.mat'
        hyperparams['labelsFile_'+str(x)] = rootDir+'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
        hyperparams['syntheticDatasetDir_'+str(x)] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
        hyperparams['cvPartitionFile_'+str(x)] = rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat'
        hyperparams['sessionName_'+str(x)] = dataDirs[x]

  
    #this seed is set for numpy and tensorflow when the class is initialized                             
    hyperparams['seed'] = datetime.now().microsecond

    return hyperparams




def getHandwritingCharacterDefinitions():
    """
    Returns a dictionary with entries that define the names of each character, its length, and where the pen tip begins.
    
    Returns:
        charDef (dict)
    """
        
    charDef = {}
    
    #Define the list of all 31 characters and their names.
    charDef['charList'] = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                'greaterThan','comma','apostrophe','tilde','questionMark']
    charDef['charListAbbr'] = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                '>',',',"'",'~','?']

    #Define the length of each character (in # of 10 ms bins) to use for each template.
    #These were hand-defined based on visual inspection of the reconstructed pen trajectories.
    charDef['charLen'] = np.array([99, 91, 70, 104, 98, 125, 110, 104, 79, 92, 127, 68, 132, 90, 
                        84, 113, 104, 74, 86, 110, 86, 83, 110, 103, 115, 100, 82, 77, 116, 71, 110]).astype(np.int32)
    
    #For each character, this defines the starting location of the pen tip (0 = bottom of the line, 1 = top)
    charDef['penStart'] = [0.25, 1, 0.5, 0.5, 0.25, 1.0, 0.25, 1.0, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 1, 
           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 1, 0.5, 1]
    
    #dictionary to convert string representation to character index
    charDef['strToCharIdx'] = {}
    for x in range(len(charDef['charListAbbr'])):
        charDef['strToCharIdx'][charDef['charListAbbr'][x]] = x
        
    #ordering of characters that kaldi (i.e., the language model) expects
    charDef['kaldiOrder'] = ['<ctc>','>',"'",',','.','?','a','b','c','d','e','f','g','h','i','j',
                             'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    #re-indexing to match kaldi order (e.g., outputs[:,:,charDef['idxToKaldi']] places the output in kald-order)
    charDef['idxToKaldi'] = np.array([31,26,28,27,29,30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                     21,22,23,24,25]).astype(np.int32)
    
    return charDef




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




def gaussianSmoothing(data, kernelSD):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.
    """

    inp = np.zeros([100])
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)

    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx].flatten()
    gaussKernel = torch.from_numpy(gaussKernel / np.sum(gaussKernel)).float()
    gaussKernel = gaussKernel.view(1, 1, -1)

    # Initialize output tensor
    smoothedData = torch.zeros_like(data)
    
    # Apply convolution for each feature across the batch
    for i in range(data.shape[1]):  # Loop over each feature channel N
        feature_data = data[:, i, :].unsqueeze(1)  # Reshape to [B, 1, T]
        
        smoothed_feature = torch.nn.functional.conv1d(feature_data, gaussKernel, padding=(gaussKernel.shape[-1] - 1) // 2)
        
        smoothedData[:, i, :] = smoothed_feature.squeeze(1)

    return smoothedData

    return smoothedData




def extractBatch(trial_iter, hyperparams, device):

    neural_data = trial_iter['neuralData']
    neural_data = neural_data.permute(0, 2, 1)
    if hyperparams['smoothInputs']:
        neural_data = gaussianSmoothing(neural_data, kernelSD=4/hyperparams['rnnBinSize'])
    neural_data = neural_data.to(device)

    targets = trial_iter['targets']
    targets = targets.permute(0, 2, 1)
    targets = targets.to(device)

    errWeights = trial_iter['errWeights']
    errWeights = errWeights.to(device)

    return neural_data, targets, errWeights




def computeFrameAccuracy(snnOutput, targets, errWeight, outputDelay):
    """
    Computes a frame-by-frame accuracy percentage given the snnOutput and the targets, while ignoring
    frames that are masked-out by errWeight and accounting for the SNN's outputDelay.
    Dimensios are swapped compared to the original function (N x T x C) --> (N x C x T)
    """
    # The best class is the one with the highest probability
    bestClass = np.argmax(snnOutput[:, 0:-1, outputDelay:], axis=1)
    indicatedClass = np.argmax(targets[:,0:-1, 0:-outputDelay], axis=1)
    bw = errWeight[:,0:-outputDelay]

    # Mean accuracy is computed by summing number of accurate frames and dividing by total number of valid frames (where bw == 1)
    acc = np.sum(bw*np.equal(np.squeeze(bestClass), np.squeeze(indicatedClass)))/np.sum(bw)
    
    return acc




def modelComplexity(hyperparam):
    # Computets the number of MACs and ACs operations of the model, accounting for neuron firing rates

    # Get the neuron firing rates
    neuron_rates = torch.load(f"{hyperparam['results_dir']}SNN_spike_rate_{hyperparam['id']}.pth")

    MACs = 0
    ACs = 0
    num_layers = hyperparam['layers']

    for layer in range(num_layers + 1):
        if layer == 0:
            # Input layer -- inputs are analog
            M = hyperparam['n_channels']
            N = hyperparam['neuron_count']
            Fin = 1
            Fr= neuron_rates[layer]
            MACs += (1 + M)*N
            ACs += (3 + Fr + N*Fr)*N
        
        elif layer == num_layers:
            # Output layer -- nospike
            M = hyperparam['neuron_count']
            N = hyperparam['n_outputs']
            Fin = neuron_rates[layer-1]
            MACs += N
            ACs += (1 + M*Fin)*N

        else:
            # Hidden layers
            M = hyperparam['neuron_count']
            N = hyperparam['neuron_count']
            Fin = neuron_rates[layer-1]
            Fr = neuron_rates[layer]
            MACs += N
            ACs += (3 + Fr + M*Fin + N*Fr)*N

    return MACs, ACs


def save_to_csv(data, name):
    #var, trial, epoch, type
    np.savetxt(name, data, delimiter=",")




def spikeplot(spike, hyperparam):
    # spike is a tuple containing the spike counts for each layer
    # each of the spike counts is a tensor of shape: (B x N)
    layers = len(spike)
    spikes = []
    for i in range(layers) :
        spikes.append(spike[i][:,:].detach().to('cpu').numpy()) 
    
    spike_rates = []
    #plot Spike count
    logging.info('Spike counts per Inference Step and Layer:')
    for i in range(layers):
        tot_spikes = np.sum(spikes[i])
        tot_neurons = spikes[i].shape[0]*spikes[i].shape[1]

        logging.info(f"Layer {str(i+1)}| Spikes: {tot_spikes:.3f} | Never spiked: {tot_neurons - np.count_nonzero(spikes[i])} neurons out of {tot_neurons} [BxN]")
        # taking into consideration the slower layer
        if i == layers - 1:
            logging.info(f"---Avg spike count of a single neuron per time step: {tot_spikes/tot_neurons / hyperparam['train_val_timeSteps'] * hyperparam['skipLen'] * 1000:.3f} * 10-3")
            spike_rates.append(tot_spikes/tot_neurons / hyperparam['train_val_timeSteps'] * hyperparam['skipLen'])
        else:
            logging.info(f"---Avg spike count of a single neuron per time step: {tot_spikes/tot_neurons / hyperparam['train_val_timeSteps'] * 1000:.3f} * 10-3")
            spike_rates.append(tot_spikes/tot_neurons / hyperparam['train_val_timeSteps'])
    
    torch.save(spike_rates, f"{hyperparam['results_dir']}SNN_spike_rate_{hyperparam['id']}.pth")
        
    
    fig, axs = plt.subplots(1,layers,figsize = (21,7))
    for i in range(layers):
        save_to_csv(spikes[i][0], hyperparam['output_csv'] + "SNN_spike_layer_{}.csv".format(i))
    for i, ax in enumerate(axs):
        ax.hist(spikes[i][0], density=False, bins=40)  
        ax.set_ylabel('Count of Neurons')
        ax.set_xlabel('Spike Rate')
        ax.set_title('layer '+str(i+1))
    plt.savefig(hyperparam['output_plots']+'Spikerate_plot_'+str(hyperparam['id'])+'.png', transparent= False, dpi=600,bbox_inches='tight' )




def neuron_hist_plot(net, hyperparam):
    fig, axs = plt.subplots(1,6,figsize = (21,7))
    LIF_layer = (net.sp1, net.sp2, net.sp3)
    LI_layer = (net.nospike)
    save_to_csv(LI_layer.tau.to('cpu').data.numpy(), hyperparam['output_csv'] + "TAU_LI")
    for i in range(3):
        save_to_csv(LIF_layer[i].tau.to('cpu').data.numpy(), hyperparam['output_csv'] + "TAU_Layer_{}".format(i))
        axs[i*2].hist(LIF_layer[i].Vth.to('cpu').data.numpy(), density=False, bins=np.linspace(-0.1,1.1,50)) 
        axs[i*2].set_ylabel('Count of Neurons')
        axs[i*2].set_xlabel('Layer '+str(i)+' Vth')
        tau = LIF_layer[i].tau.to('cpu').data.numpy()
        axs[i*2+1].hist(LIF_layer[i].tau.to('cpu').data.numpy(), density=False, bins=np.linspace(min(tau),max(tau),50)) 
        axs[i*2+1].set_ylabel('Count of Neurons')
        axs[i*2+1].set_xlabel('Layer '+str(i)+' tau')
    plt.savefig(hyperparam['output_plots']+'Neuron_distribution_plot_'+str(hyperparam['id'])+'.png', transparent= False, dpi=600,bbox_inches='tight' )



def TrainPlot(metrics, hyperparam):

    trainloss_per_epoch = metrics['trainloss_per_epoch']
    trainacc_per_epoch = metrics['trainacc_per_epoch']
    valloss_per_epoch = metrics['valloss_per_epoch']
    valacc_per_epoch = metrics['valacc_per_epoch']


    fig, axs = plt.subplots(2,2,figsize = (21,14))
    axs[0,0].plot(trainloss_per_epoch)
    axs[0,0].set_ylabel('Loss')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_title('Train Loss per Epoch')
    axs[0,1].plot(trainacc_per_epoch, color='orange')
    axs[0,1].set_ylabel('Accuracy')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_title('Train Accuracy per Epoch')
    axs[1,0].plot(valloss_per_epoch)
    axs[1,0].set_ylabel('Loss')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_title('Validation Loss per Epoch')
    axs[1,1].plot(valacc_per_epoch, color='orange')
    axs[1,1].set_ylabel('Accuracy')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_title('Validation Accuracy per Epoch')
    plt.savefig(hyperparam['output_plots']+'Train_plot_'+str(hyperparam['id'])+'.png', transparent= False, dpi=600,bbox_inches='tight' )




def trainModel(model, train_loader, optimizer, scheduler, criterion, hyperparams, device):
    model.train()
    num_batches = len(train_loader)

    running_loss = []
    running_acc = []
    train_progress = "Train progress: |"
    update_freq = 10 # fraction of batches before updating the progress bar

    for i, trial_iter in enumerate(train_loader):

        data, targets, errWeights = extractBatch(trial_iter, hyperparams, device)
        optimizer.zero_grad()
        if hyperparams['network_type'] != 'RNN':
            output, spikecounts = model(data)
        else:
            output = model(data)
        loss = criterion(output, targets, errWeights)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if hyperparams['scheduler'] == 'LambdaLR' or hyperparams['scheduler'] == 'StepLR':
            scheduler.step()
        running_loss.append(loss.item())

        outputs, targets, errWeights = tensors_to_numpy(output, targets, errWeights)
        acc = computeFrameAccuracy(outputs, targets, errWeights, hyperparams['outputDelay'])
        running_acc.append(acc)

        #print(f"{train_progress} Batch: {i+1}/{num_batches} | Loss: {loss.item():.3f}")

    if hyperparams['scheduler'] == 'ReduceLROnPlateau':
        scheduler.step(sum(running_acc) / len(running_acc))

    #print("")

    return running_loss, running_acc




def tensors_to_numpy(data, targets, errWeights):
    data = data.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    errWeights = errWeights.cpu().detach().numpy()

    return data, targets, errWeights




def validateModel(model, val_loader, criterion, hyperparams, device):
    model.eval()
    num_batches = len(val_loader)

    running_loss = []
    running_acc = []
    val_progress = "Validation progress: |"
    update_freq = 10 # fraction of batches before updating the progress bar

    with torch.no_grad():

        for i, trial_iter in enumerate(val_loader):

            data, targets, errWeights = extractBatch(trial_iter, hyperparams, device)
            if hyperparams['network_type'] != 'RNN':
                output, spikecounts = model(data)
            else:
                output = model(data)
            loss = criterion(output, targets, errWeights)
            running_loss.append(loss.item())
            output, targets, errWeights = tensors_to_numpy(output, targets, errWeights)
            acc = computeFrameAccuracy(output, targets, errWeights, hyperparams['outputDelay'])
            running_acc.append(acc)
    
            #print(f"{val_progress} Batch: {i+1}/{num_batches} | Loss: {loss.item():.3f}")

    #print("")
    if hyperparams['network_type'] != 'RNN':
        spikeplot(spikecounts, hyperparams)

    return running_loss, running_acc

    


def testModel(model, test_loaders, viable_test_days ,criterion, hyperparams, device):

    charDef = getHandwritingCharacterDefinitions()
    allErrCounts = []
    predictions = {}

    model.eval()

    with torch.no_grad():

        for idx_loader in range(len(test_loaders)):

            print(f"Testing on day {hyperparams['dataDirs'][viable_test_days[idx_loader]]}")

            test_loader = iter(test_loaders[idx_loader])
            num_batches = len(test_loaders[idx_loader])

            running_loss = 0.0
            running_acc = 0.0
            test_progress = "Testing progress: |"

            outputs = []

            for i in range(num_batches):

                trial_iter = next(test_loader)
                data, targets, errWeights = extractBatch(trial_iter, hyperparams, device)
                output = model(data)
                loss = criterion(output, targets, errWeights)
                running_loss += loss.item()
                output, targets, errWeights = tensors_to_numpy(output, targets, errWeights)
                acc = computeFrameAccuracy(output, targets, errWeights, hyperparams['outputDelay'])
                running_acc += acc

                #if i%(np.ceil(num_batches/100))==0:
                test_progress += "#"
                print(f"{test_progress} {loss.item():.4f}", end='\r')

                outputs.append(output)
            

            outputs = np.concatenate(outputs, axis=0)
            num_sentences = outputs.shape[0]
            print(f"Number of sentences of the day: {num_sentences}")
            predictions[hyperparams['dataDirs'][viable_test_days[idx_loader]]] = outputs
            

            # Character error rate and Word error rate computation
            dayIdx = viable_test_days[idx_loader]
            cvPartFile = scipy.io.loadmat(hyperparams['cvPartitionFile_'+str(dayIdx)])
            testIdx = cvPartFile[hyperparams['dataDirs'][dayIdx]+'_test']
                                                    
            sentenceDat = scipy.io.loadmat(hyperparams['sentencesFile_'+str(dayIdx)])

            errCounts, decSentences = evaluateSNNOutput(outputs, sentenceDat['numTimeBinsPerSentence'][testIdx] / hyperparams['rnnBinSize'] + hyperparams['outputDelay'],
                                                        sentenceDat['sentencePrompt'][testIdx],
                                                        charDef,
                                                        charStartThresh=0.3,
                                                        charStartDelay=15)
            
            #save decoded sentences, character error rates and word error rates for later summarization
            saveDict = {}
            saveDict['decSentences'] = decSentences
            saveDict['trueSentences'] = sentenceDat['sentencePrompt'][testIdx]
            saveDict.update(errCounts)
                    
            valAcc = 100*(1 - np.sum(errCounts['charErrors']) / np.sum(errCounts['charCounts']))

            print('Character error rate for this session: %1.2f%%' % float(100-valAcc))
            print('Below is the decoder output for all validation sentences in this session:')
            print(' ')
            
            for v in np.arange(num_sentences):
                trueText = sentenceDat['sentencePrompt'][testIdx][v,0][0]
                trueText = trueText.replace('>',' ')
                trueText = trueText.replace('~','.')
                trueText = trueText.replace('#','')
                
                print('#' + str(v) + ':')
                print('True:    ' + trueText)
                print('Decoded: ' + decSentences[v])
                print(' ')
        
            #put together all the error counts from all sessions so we can compute overall error rates below
            allErrCounts.append(np.stack([errCounts['charCounts'],
                                    errCounts['charErrors'],
                                    errCounts['wordCounts'],
                                    errCounts['wordErrors']],axis=0).T)
        
    
    # Save predicitons
    predictions_file = 'inferOutputs' + 'predictions.mat'
    scipy.io.savemat(predictions_file, predictions)

    # Summarize character error rate and word error rate across all sessions
    concatErrCounts = np.squeeze(np.concatenate(allErrCounts, axis=0))
    cer = 100*(np.sum(concatErrCounts[:,1]) / np.sum(concatErrCounts[:,0]))
    wer = 100*(np.sum(concatErrCounts[:,3]) / np.sum(concatErrCounts[:,2]))

    print('Character error rate: %1.2f%%' % float(cer))
    print('Word error rate: %1.2f%%' % float(wer))



def trainModel_Inf(num_batches, model, train_iterators, viable_train_days, val_iterators, viable_val_days, optimizer, scheduler, criterion, hyperparams, device):

    running_train_loss = []
    running_train_acc = []
    running_val_loss = []
    running_val_acc = []
    update_freq = 10 # number of batches before updating the progress bar
    progress = "Batch progress: |"

    for i in range(num_batches):

        model.train()
        random_day = np.random.choice(np.arange(len(viable_train_days)))
        trial_iter = train_iterators.getNextIter(random_day)
        data, targets, errWeights = extractBatch(trial_iter, hyperparams, device)
        optimizer.zero_grad()
        output, spikecounts = model(data)
        loss = criterion(output, targets, errWeights)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_train_loss.append(loss.item())
        output, targets, errWeights = tensors_to_numpy(output, targets, errWeights)
        train_acc = computeFrameAccuracy(output, targets, errWeights, hyperparams['outputDelay'])
        running_train_acc.append(train_acc)

        progress += "#"
        if i > 0: print(progress, end='\r')

        if i % update_freq == 0:
            print("")
            progress = "Batch progress: |"
            print(f"Batch {i+1}/{num_batches} | Training loss: {loss.item():.4f} | Training accuracy: {train_acc:.4f}")
            logging.info(f"Batch {i+1}/{num_batches} | Training loss: {loss.item():.4f} | Training accuracy: {train_acc:.4f}")

        if i % hyperparams['updateVal'] == 0:
            model.eval()
            with torch.no_grad():
                for i in range(hyperparams['batchesPerVal']):
                    # get index of a random day
                    random_val_day = np.random.choice(np.arange(len(viable_val_days)))
                    val_iter = val_iterators.getNextIter(random_val_day)
                    data, targets, errWeights = extractBatch(val_iter, hyperparams, device)
                    output, spikecounts = model(data)
                    val_loss = criterion(output, targets, errWeights)
                    output, targets, errWeights = tensors_to_numpy(output, targets, errWeights)
                    val_acc = computeFrameAccuracy(output, targets, errWeights, hyperparams['outputDelay'])

                    running_val_loss.append(val_loss.item())
                    running_val_acc.append(val_acc)

            avg_val_loss = np.sum(running_val_loss[-hyperparams['batchesPerVal']:])/hyperparams['batchesPerVal']
            avg_val_acc = np.sum(running_val_acc[-hyperparams['batchesPerVal']:])/hyperparams['batchesPerVal']
            print(f"Validation loss: {avg_val_loss:.4f} | Validation accuracy: {avg_val_acc:.4f}")
            logging.info(f"Validation loss: {avg_val_loss:.4f} | Validation accuracy: {avg_val_acc:.4f}")

    return running_train_loss, running_train_acc, running_val_loss, running_val_acc



        
        





    



        















