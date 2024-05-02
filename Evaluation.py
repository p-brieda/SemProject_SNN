import scipy.special
import numpy as np

def evaluateSNNOutput(snnOutput, numBinsPerSentence, trueText, charDef, charStartThresh=0.3, charStartDelay=15):
    """
    Converts the rnn output (character probabilities & a character start signal) into a discrete sentence and computes
    char/word error rates. Returns error counts and the decoded sentences.
    """  
    lgit = snnOutput[:,0:-1,:]
    charStart = snnOutput[:,-1,:]

    #convert output to character strings
    decStr = decodeCharStr(lgit, charStart, charStartThresh, charStartDelay, 
                           numBinsPerSentence, charDef['charListAbbr'])

    allErrCounts = {}
    allErrCounts['charCounts'] = np.zeros([len(trueText)])
    allErrCounts['charErrors'] = np.zeros([len(trueText)])
    allErrCounts['wordCounts'] = np.zeros([len(trueText)])
    allErrCounts['wordErrors'] = np.zeros([len(trueText)])
    
    allDecSentences = []

    #compute error rates
    for t in range(len(trueText)):
        thisTrueText = trueText[t,0][0]
        thisTrueText = thisTrueText.replace(' ','')
        thisTrueText = thisTrueText.replace('>',' ')
        thisTrueText = thisTrueText.replace('~','.')
        thisTrueText = thisTrueText.replace('#','')

        thisDec = decStr[t]
        thisDec = thisDec.replace('>',' ')
        thisDec = thisDec.replace('~','.')

        nCharErrors = wer(list(thisTrueText), list(thisDec))
        nWordErrors = wer(thisTrueText.strip().split(), thisDec.strip().split())
        
        allErrCounts['charCounts'][t] = len(thisTrueText)
        allErrCounts['charErrors'][t] = nCharErrors
        allErrCounts['wordCounts'][t] = len(thisTrueText.strip().split())
        allErrCounts['wordErrors'][t] = nWordErrors

        allDecSentences.append(thisDec)

    return allErrCounts, allDecSentences




def decodeCharStr(logitMatrix, transSignal, transThresh, transDelay, numBinsPerTrial, charList):
    """
    Converts the rnn output (character probabilities & a character start signal) into a discrete sentence.
    """
    decWords = []
    for v in range(logitMatrix.shape[0]):
        logits = np.squeeze(logitMatrix[v,:,:])
        bestClass = np.argmax(logits, axis=0)
        letTrans = scipy.special.expit(transSignal[v,:])

        endIdx = np.ceil(numBinsPerTrial[v]).astype(int)
        letTrans = letTrans[0:endIdx[0]]

        transIdx = np.argwhere(np.logical_and(letTrans[0:-1]<transThresh, letTrans[1:]>transThresh))
        transIdx = transIdx[:,0]
        
        wordStr = ''
        for x in range(len(transIdx)):
            wordStr += charList[bestClass[transIdx[x]+transDelay]]

        decWords.append(wordStr)
        
    return decWords





def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]