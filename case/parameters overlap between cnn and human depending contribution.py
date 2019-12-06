# Calculate the overlap of high informative parameters between cnn and human

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def searchHighContriParam(paramD,threshold=0.8):
    # calculate the sum of contribution and calculate the contribution percentage of signal parameter.
    sumContrib = paramD.sum()
    percentContrib = paramD / sumContrib

    # sort the contribution and calculate the cumulative contribution along the decreasing with contribution.
    paramContribSortIndex = np.argsort(-percentContrib)
    paramContribSorted = percentContrib[np.argsort(-percentContrib)]
    cumSumContrib = np.cumsum(paramContribSorted)

    # find the parameters which occupy 80% contribution and more informative than others.
    # return the high informative parameters Index
    paramOverTh = len(np.argwhere(cumSumContrib < threshold))+1
    paramIndexOverTh = paramContribSortIndex[:paramOverTh]
    return paramIndexOverTh


def paramScale(param):
    ParamScale = []
    ParamScale.append(len(param[param < 12]))
    ParamScale.append(len(param[(param >= 12) & (param < 60)]))
    ParamScale.append(len(param[(param >= 60) & (param < 252)]))
    ParamScale.append(len(param[(param >= 252) & (param < 1020)]))
    ParamScale.append(len(param[(param >= 1020) & (param < 4092)]))
    return ParamScale


# load the overlap parameters between cnn and human
overlapParam = np.load(r'D:\cnnface\gender_analysis\Result\difference/human2cnnOverlap.npy')
diffParam = np.load(r'D:\cnnface\gender_analysis\Result\difference/human2cnnDiff.npy')

paramNumScale = np.array([12, 48, 192, 768, 3072])
overlapParamScale = np.array(paramScale(overlapParam))/paramNumScale
diffParamScale = np.array(paramScale(diffParam))/paramNumScale

scale = [2, 4, 8, 16, 32]
sns.barplot(scale, overlapParamScale)
plt.title('Overlap Parameters')
plt.show()
sns.barplot(scale, diffParamScale)
plt.title('Different Parameters')
plt.show()
sns.barplot(scale, diffParamScale + overlapParamScale)
plt.title('High Contribution parameters')
plt.show()
