# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:02:47 2019

@author: qyk
"""
# tidy data
import numpy as np
import pandas as pd 

d_cnn = np.load(r'C:\Myfile\File\Project\AI\similarity of parameter contribution/d_cnn.npy')
d_human = np.load(r'C:\Myfile\File\Project\AI\similarity of parameter contribution/d_human.npy')

cnnContri = d_cnn/d_cnn.sum()
humanContri = d_human/d_human.sum()

sumContrib_cnn = d_cnn.sum()
percentContrib_cnn = d_cnn / sumContrib_cnn

# sort the contribution and calculate the cumulative contribution along the decreasing with contribution.
paramContribSortIndex = np.argsort(-percentContrib_cnn)
paramContribSorted_cnn = percentContrib_cnn[np.argsort(-percentContrib_cnn)]
cumSumContrib_cnn = np.cumsum(paramContribSorted_cnn)

# find the parameters which occupy 80% contribution and more informative than others.
# return the high informative parameters Index
paramOver = len(np.argwhere(cumSumContrib_cnn < 0.8))+1
paramIndexOver = paramContribSortIndex[:paramOver]

over80 = np.zeros(4092)
over80[paramIndexOver] = 1

df = pd.DataFrame({'d_cnn':d_cnn,
                  'contribution_cnn':cnnContri,
                  'd_human':d_human,
                  'contribution_human':humanContri,
                  'Over80':over80})
df.to_csv(r'C:\Myfile\File\Project\AI\similarity of parameter contribution/data.csv')


# calculate the contribution of the 1995 parameters in different scale
IndexOver80 = df.index[df.Over80==1]