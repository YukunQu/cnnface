# Calculate the contribution of high informative parameters of cnn for human

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load the conhens'd of parameters in cnn
d_cnn = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/cohensd_sum.npy')

# calculate the sum of contribution and calculate the contribution percentage of signal parameter.
sumContrib = d_cnn.sum()
percentContrib = d_cnn / sumContrib

# sort the contribution and calculate the cumulative contribution along the decreasing with contribution.
paramContribSortIndex = np.argsort(-percentContrib)
paramContribSorted = percentContrib[np.argsort(-percentContrib)]
cumSumContrib = np.cumsum(paramContribSorted)

# plot the cumulative contribution line chart.
sns.set_style('darkgrid')
bins = range(len(paramContribSorted))
sns.lineplot(bins,paramContribSorted)
plt.xlabel('parameter')
plt.xticks([0,400,1000,2000,3000,4000])
plt.ylabel('Contribution Ratio')
plt.show()
sns.lineplot(bins,cumSumContrib)
plt.xlabel('parameter')
plt.ylabel('Cumulative contribution')
plt.show()

# find the parameters which occupy 80% contribution and more informative than others.
# return the high informative parameters Index
paramOver80 = len(np.argwhere(cumSumContrib < 0.8))+1
paramIndexOver80 = paramContribSortIndex[:paramOver80]

# load the the conhens'd of parameters in human
d_human = np.load(r'D:\cnnface\gender_analysis\human_result\para_significant/cohensd_sum.npy')

# calculate the sum of contribution and calculate the contribution percentage of signal parameter.(human)
sumContrib_human = d_human.sum()
percentContrib_human = d_human / sumContrib_human

# Calculate the contribution of high informative parameters of cnn for human using the high informative parameters Index
cnnParamContrib2human = percentContrib_human[paramIndexOver80]

# sum up the contribution percentages of the parameters
cnnParamContrib2humanSum = cnnParamContrib2human.sum()

# sum up the contribution percentages of the parameters in different frequency scales
humanContribScale = []
humanContribScale.append(np.sum(percentContrib_human[:12]))
humanContribScale.append(np.sum(percentContrib_human[12:60]))
humanContribScale.append(np.sum(percentContrib_human[60:252]))
humanContribScale.append(np.sum(percentContrib_human[252:1020]))
humanContribScale.append(np.sum(percentContrib_human[1020:4092]))

cnnContrib2humanScale = []
cnnContrib2humanScale.append(np.sum(percentContrib_human[paramIndexOver80[paramIndexOver80<12]]))
cnnContrib2humanScale.append(np.sum(percentContrib_human[paramIndexOver80[(paramIndexOver80>=12) & (paramIndexOver80<60)]]))
cnnContrib2humanScale.append(np.sum(percentContrib_human[paramIndexOver80[(paramIndexOver80>=60) & (paramIndexOver80<252)]]))
cnnContrib2humanScale.append(np.sum(percentContrib_human[paramIndexOver80[(paramIndexOver80>=252) & (paramIndexOver80<1020)]]))
cnnContrib2humanScale.append(np.sum(percentContrib_human[paramIndexOver80[(paramIndexOver80>=1020) & (paramIndexOver80<4092)]]))

category = [2,4,8,16,32]
sns.barplot(category,humanContribScale)
plt.xlabel('Scale')
plt.ylabel('Contribution Ratio')
plt.show()
sns.barplot(category,cnnContrib2humanScale)
plt.xlabel('Scale')
plt.ylabel('Contribution Ratio')
plt.show()

contriRatioScale = np.array(cnnContrib2humanScale)/np.array(humanContribScale)
sns.barplot(category,contriRatioScale)
plt.xlabel('Scale')
plt.ylabel('Contribution Ratio')
plt.show()


# calculate the overlap of parameters occupied 80% contribution between cnn and human from 0 ~ 100.


# get paramIndexOver80 and paramIndexOver80Human


