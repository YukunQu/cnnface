import numpy as np
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
#import scipy.stats as stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from cnnface.stimulus.Image_process import nor
from cnnface.analysis.generate_ci import generateCI,cal_paramci
import itertools
import seaborn as sns
sns.set_style('darkgrid')

# generate the all CIs from human experiment data
label_exp = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
param_exp = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
subID = list(range(1,17))
idCom_all = list(itertools.combinations(subID,1))
idComExcept = [list(set(subID).difference(set(id))) for id in idCom_all]
ci_com = []
for id_com in idCom_all:
    param_ci = cal_paramci(param_exp,label_exp,id_com)
    ci = nor(generateCI(param_ci))
    ci_com.append(ci.reshape(-1))

ci_comExcept = []
for id_comexc in idComExcept:
    param_ci = cal_paramci(param_exp,label_exp,id_comexc)
    ci = nor(generateCI(param_ci))
    ci_comExcept.append(ci.reshape(-1))

# generate the 120 CIs from cnn trials
label_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
param_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
ci_cnn = []
for i in range(len(idCom_all)):
    indexList = np.arange(0,19999)
    randomIndex = np.random.choice(indexList, 1000)
    label_rand = label_cnn[randomIndex]
    param_rand = param_cnn[randomIndex]
    param_ci = cal_paramci(param_rand,label_rand)
    ci = nor(generateCI(param_ci))
    ci_cnn.append(ci.reshape(-1))

ci_all = np.array(ci_com + ci_comExcept + ci_cnn)

pca_pixel = PCA(32)
pca_pixel.fit(ci_all)
varRatio = pca_pixel.explained_variance_ratio_
bins = range(len(varRatio))
sns.lineplot(bins,varRatio)
plt.xlabel('Component')
plt.ylabel('VarianceRatio')
plt.xticks(range(0,len(varRatio),3))
plt.show()