# Independent t test for 4092 parameters

import numpy as np
from scipy import stats
import statsmodels.api as sm

params = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
label = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')

label_0 = np.argwhere(label == 0).astype('int32')
label_1 = np.argwhere(label == 1).astype('int32')

s_sum = []
p_sum = []
dis_sum = []
for i in range(4092):
    x_5000 = params[:, i]
    x_0 = x_5000[label_0]
    x_1 = x_5000[label_1]

    s, p = stats.ttest_ind(x_0, x_1)
    s_sum.append(s)
    p_sum.append(p)

    d = lambda x1,x2: (x1.mean() - x2.mean()) / np.sqrt(((np.std(x1))**2 + (np.std(x2))**2)/2)
    dis = np.abs(d(x_0,x_1))
    dis_sum.append(dis)

p_sum = np.squeeze(np.array(p_sum))
s_sum = np.squeeze(np.array(s_sum))
p_sum_sign = p_sum[p_sum < (0.05/4092)]
#p_signIndex = np.squeeze(np.argwhere(p_sum < (0.05/4092)))
p_signIndexFDR = sm.stats.multipletests(p_sum,alpha=0.05,method='fdr_bh')
p_minsignIndex = np.squeeze(np.argwhere(p_sum == p_sum_sign.max()))

np.save(r'D:\cnnface\gender_analysis\human_result\para_significant/p_sum.npy', p_sum)
np.save(r'D:\cnnface\gender_analysis\human_result\para_significant/s_sum.npy', s_sum)
np.save(r'D:\cnnface\gender_analysis\human_result\para_significant/cohensd_sum.npy', dis_sum)
np.save(r'D:\cnnface\gender_analysis\human_result\para_significant/p_signIndexFDR.npy', p_signIndexFDR)
np.save(r'D:\cnnface\gender_analysis\human_result\para_significant/p_minsignIndex.npy', p_minsignIndex)