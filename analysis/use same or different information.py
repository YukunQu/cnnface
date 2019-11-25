import os
import numpy as np
from cnnface.stimulus.Image_process import img_similarity,nor
from cnnface.analysis.generate_ci import generateCI, cal_ci
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools


# generate the 120 CIs from human experiment data
label_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/label_sum.npy')
param_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/param_exp.npy')
subID = list(range(1,17))
idCom_all = list(itertools.combinations(subID,5))
idComExcept = [list(set(subID).difference(set(id))) for id in idCom_all]
ci_com = []
for id_com in idCom_all:
    param_ci = cal_ci(param_exp,label_exp,id_com)
    ci = nor(generateCI(param_ci))
    ci_com.append(ci)


ci_comExcept = []
for id_comexc in idComExcept:
    param_ci = cal_ci(param_exp,label_exp,id_comexc)
    ci = nor(generateCI(param_ci))
    ci_comExcept.append(ci)


# generate the 120 CIs from cnn trials
label_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
param_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
ci_cnn = []
for i in range(len(idCom_all)):
    indexList = np.arange(0,19999)
    randomIndex = np.random.choice(indexList, 5000)
    label_rand = label_cnn[randomIndex]
    param_rand = param_cnn[randomIndex]
    param_ci = cal_ci(param_rand,label_rand)
    ci = nor(generateCI(param_ci))
    ci_cnn.append(ci)

human2human = [img_similarity(com, exc, 'pearson') for com,exc in zip(ci_com,ci_comExcept)]
cnn2human = [img_similarity(cnn, exc, 'pearson') for cnn,exc in zip(ci_cnn,ci_comExcept)]

sns.set_style('darkgrid')
sns.distplot(human2human,label='human')
sns.distplot(cnn2human,label='cnn')
plt.show()

s, p = stats.ttest_ind(human2human,cnn2human)