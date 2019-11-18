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

from cnnface.analysis.generate_ci import generateCI,cal_ci

# generate the 100 CIs from human experiment data
label_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/label_sum.npy')
param_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/param_expxx.npy')
ci_exp = []
for i in range(100):
    indexList = np.arange(0,9999)
    randomIndex = np.random.choice(indexList, 5000)
    label_rand = label_exp[randomIndex]
    param_rand = param_exp[randomIndex]
    param_ci = cal_ci(param_rand,label_rand)
    ci = generateCI(param_ci)
    ci_exp.append(ci.reshape(-1))

# generate the 100 CIs from CNN experiment data
label_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
param_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
ci_cnn = []
for i in range(100):
    indexList = np.arange(0,19999)
    randomIndex = np.random.choice(indexList, 5000)
    label_rand = label_cnn[randomIndex]
    param_rand = param_cnn[randomIndex]
    param_ci = cal_ci(param_rand,label_rand)
    ci = generateCI(param_ci)
    ci_cnn.append(ci.reshape(-1))

# generate the 100 random CIs
ci_rand = []
for i in range(100):
    label_rand = np.random.choice((0,1),5000)
    param_rand = param_exp[:5000,:]
    param_ci = cal_ci(param_rand, label_rand)
    ci = generateCI(param_ci)
    ci_rand.append(ci.reshape(-1))

# combine data
ci_all = np.array(ci_exp + ci_cnn + ci_rand)

#compute t-SNE visulization
from sklearn import manifold
tsne = manifold.TSNE(n_components=2, learning_rate=50, n_iter=5000, init='pca', random_state=500)
ac_tsne = tsne.fit_transform(ci_all)

tsne_a = ac_tsne[:100,:]
tsne_c = ac_tsne[100:200,:]
tsne_b = ac_tsne[200:300,:]

#plot 2D picture
ax = plt.figure(figsize=(8, 6), dpi=100, facecolor='grey', edgecolor = 'grey') #,facecolor='dimgray', edgecolor = 'dimgray' facecolor='grey',
plt.rcParams['axes.facecolor'] = 'gray'

plt.scatter(tsne_a[:,0], tsne_a[:,1], s=40, c="yellow", label='Asian', alpha=1, linewidths=0)
plt.scatter(tsne_c[:,0], tsne_c[:,1], s=40, c="white", label="Caucasian", alpha=1, linewidths=0) #aliceblue, linewidths=None
plt.scatter(tsne_b[:,0], tsne_b[:,1], s=30, c="black", label="Black", alpha=1, linewidths=0)
#ax1.set_facecolor('grey')
#ax1 = plt.gca()
#ax1.set_xlabel('PC1', fontdict={'size': 12, 'color': 'black'})
#ax1.set_ylabel('PC2', fontdict={'size': 12, 'color': 'black'})

plt.title('Visualize 3 races distribution in fc2 using t-SNE')
plt.legend()
#plt.legend(loc='best')
plt.show()

'''
#plot 3D picture
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6), dpi=100)
ax = Axes3D(fig)
#ax.scatter(tsne_a[:,0], tsne_a[:,1], tsne_a[:,2], c="red", label='Asian')
#ax.scatter(tsne_c[:,0], tsne_c[:,1], tsne_c[:,2], c="black", label="Caucasian")
ax.scatter(tsne_b[:,0], tsne_b[:,1], tsne_b[:,2], c="orange", label="Black")

ax.legend()
#ax.legend(loc='best')
ax.set_xlabel(fontdict={'size': 12, 'color': 'black'})
ax.set_ylabel(fontdict={'size': 12, 'color': 'black'})
ax.set_zlabel(fontdict={'size': 12, 'color': 'black'})
plt.show()
'''