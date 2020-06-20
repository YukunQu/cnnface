import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Calculate the contribution of high informative parameters of cnn for human

# load the conhens'd of parameters in cnn
d_cnn = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/d_abs.npy')

# load the the conhens'd of parameters in human
d_human = np.load(r'D:\cnnface\gender_analysis\human_result\para_significant/cohensd_sum.npy')

def cnnParamContrib2humanSum(d_cnn,d_human,threshold):
    # calculate the sum of contribution and calculate the contribution percentage of signal parameter.
    sumContrib_cnn = d_cnn.sum()
    percentContrib_cnn = d_cnn / sumContrib_cnn

    # sort the contribution and calculate the cumulative contribution along the decreasing with contribution.
    paramContribSortIndex = np.argsort(-percentContrib_cnn)
    paramContribSorted_cnn = percentContrib_cnn[np.argsort(-percentContrib_cnn)]
    cumSumContrib_cnn = np.cumsum(paramContribSorted_cnn)

    # find the parameters which occupy 80% contribution and more informative than others.
    # return the high informative parameters Index
    paramOver = len(np.argwhere(cumSumContrib_cnn < threshold))+1
    paramIndexOver = paramContribSortIndex[:paramOver]

    # calculate the sum of contribution and calculate the contribution percentage of signal parameter.(human)
    sumContrib_human = d_human.sum()
    percentContrib_human = d_human / sumContrib_human

    # Calculate the contribution of high informative parameters of cnn for human using the high informative parameters Index
    cnnParamContrib2human = percentContrib_human[paramIndexOver]

    # sum up the contribution percentages of the parameters
    cnnParamContrib2humanSum = cnnParamContrib2human.sum()

    np.save(r'D:\cnnface\Data_sorted\compareResult\contribution/d_cnn_ContriOver80Index.npy',paramIndexOver)
    x= d_cnn[paramIndexOver]
    print(len(x))
    y= d_human[paramIndexOver]
    paramContriSimilarity = stats.pearsonr(x,y)

    return cnnParamContrib2humanSum, paramContriSimilarity

threshold = np.arange(0.01,0.91,0.01)
cnnParamContrib2humanAll = []
similarityAll = []
for th in threshold:
    contribution, similarity = cnnParamContrib2humanSum(d_cnn, d_human, th)
    cnnParamContrib2humanAll.append(contribution)
    similarityAll.append(similarity[0])

sns.set_style('darkgrid')
sns.lineplot(threshold,cnnParamContrib2humanAll)
plt.xlabel('Contribution for CNN')
plt.ylabel('Contribution for Human')
plt.title('The contribution of high informative parameters in cnn for human')
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution/ContritbuionOfCNNforhuman.png',dpi=300)
plt.show()

sns.lineplot(threshold[-80:],similarityAll[-80:])
plt.xlabel('High contribution parameter of CNN')
plt.ylabel('Similarity')
plt.title('Similarity of parameters contribution between human and CNN ')
plt.show()