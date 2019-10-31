# Independent t test for 4092 parameters

import numpy as np
from scipy import stats

params_5000 = np.load(r'D:\cnnface\Emotion_analysis\noise_metadata/neu_params_5000.npy')

label_0 = np.loadtxt('D:\cnnface\Emotion_analysis\CI_analysis/neu_label_happy.txt').astype('int64')
label_1 = np.loadtxt('D:\cnnface\Emotion_analysis\CI_analysis/neu_label_sad.txt').astype('int64')

s_sum = []
p_sum = []
for i in range(4092):
    x_5000 = params_5000[:, i]
    x_f = x_5000[label_0]
    x_m = x_5000[label_1]

    s, p = stats.ttest_ind(x_f, x_m)
    s_sum.append(s)
    p_sum.append(p)


p_sum = np.array(p_sum)
s_sum = np.array(s_sum)
p_sum_sign = p_sum[p_sum < (0.05/4092)]
p_signIndex = np.squeeze(np.argwhere(p_sum < (0.05/4092)))
p_minsignIndex = np.argwhere(p_sum == p_sum_sign.max())

np.save(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/p_signIndex.npy', p_signIndex)
np.save(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/p_sum.npy', p_sum)
np.save(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/s_sum.npy', s_sum)
np.save(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/p_minsignIndex.npy',p_minsignIndex)