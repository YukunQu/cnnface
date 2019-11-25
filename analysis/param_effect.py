# ------------------------------------------------------------------------------ #
# Check how the single parameter effect the classification probability of vgg    #
# ------------------------------------------------------------------------------ #
import numpy as np
import pandas as pd


# load classification probability of baseFace
baseface_prob = np.load(r'D:\cnnface\gender_analysis\CI_analysis\param_effect/baseface_prob.npy')
param_prob = np.load(r'D:\cnnface\gender_analysis\CI_analysis\param_effect/gender_label_pro_singlePara.npy')

p_signIndex = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/p_signIndex.npy')
condition = pd.read_csv(r'D:\cnnface\gender_analysis\CI_analysis\param_effect/single_param_img.csv', skiprows=1)['condition']
condition = np.array(condition)
# calculate the probability result of baseface
dist_prob = param_prob - baseface_prob

dist_params = []
max_indexs = []
for param in p_signIndex:
    param = param + 1
    dist_singlePara = dist_prob[condition == param]
    # dist_singleIndex = condition == param
    # max_index = [dist_singlePara == dist_max]
    dist_singlePara_max = np.max(np.abs(dist_singlePara))
    dist_params.append(dist_singlePara_max)
    # max_indexs.append(max_index)
dist_params = np.array(dist_params)

np.save(r'D:\cnnface\gender_analysis\CI_analysis\param_effect/dist_params.npy', dist_params)