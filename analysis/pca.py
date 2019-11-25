# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:02:45 2019

@author: Administrator

commend PCA for activation matrix
"""
import numpy as np
import pandas as pd
#import scipy.stats as stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from cnnface.stimulus.Image_process import nor
from cnnface.analysis.generate_ci import generateCI,cal_ci
import itertools


# generate the all CIs from human experiment data
label_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/label_sum.npy')
param_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/param_exp.npy')
subID = list(range(1,17))
idCom_all = list(itertools.combinations(subID,1))
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
    param_ci = cal_ci(param_rand,label_rand)
    ci = nor(generateCI(param_ci))
    ci_cnn.append(ci)

# #generate the 120 random CIs
# ci_rand = []
# for i in range(len(idCom_all)):
#     label_rand = np.random.choice((0,1),1000)
#     param_rand = param_exp[:5000,:]
#     param_ci = cal_ci(param_rand, label_rand)
#     ci = nor(generateCI(param_ci))
#     ci_rand.append(ci.reshape(-1))


# combine data
ci_all = np.array(ci_com + ci_comExcept + ci_cnn)

#compute t-SNE visulization
from sklearn import manifold
tsne = manifold.TSNE(n_components=2, learning_rate=50, n_iter=5000, init='pca', random_state=500)
ac_tsne = tsne.fit_transform(ci_all)

ci_num = len(idCom_all)
tsne_com = ac_tsne[:ci_num, :]
tsne_comExcept = ac_tsne[ci_num:ci_num*2,:]
tsne_cnn = ac_tsne[ci_num*2:, :]

#plot 2D picture
ax = plt.figure(figsize=(8, 6), dpi=100, facecolor='white', edgecolor = 'white') #,facecolor='dimgray', edgecolor = 'dimgray' facecolor='grey',
plt.rcParams['axes.facecolor'] = 'white'

plt.scatter(tsne_com[:,0], tsne_com[:,1], s=40, c="orange", label='human', alpha=1, linewidths=0)
plt.scatter(tsne_comExcept[:,0], tsne_comExcept[:,1], s=60, c="red", label="human_template", alpha=1, linewidths=0)
plt.scatter(tsne_cnn[:,0], tsne_cnn[:,1], s=40, label="cnn", alpha=1, linewidths=0) #aliceblue, linewidths=None
#plt.scatter(tsne_rand[:,0], tsne_cnn[:,1], s=40, c='black', label="random", alpha=1, linewidths=0) #aliceblue, linewidths=None


plt.title('Visualize the ci of distribution using t-SNE')
plt.legend()
#plt.legend(loc='best')
plt.show()
