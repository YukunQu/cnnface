# Independent t test for 4092 parameters

import numpy as np
from numpy import std, mean, sqrt
from scipy import stats
import statsmodels.api as sm


def ttest_levene(d1,d2):
    s, p_var = stats.levene(d1,d2)
    if p_var > 0.05:
        t, p = stats.ttest_ind(d1,d2)
    else:
        t, p = stats.ttest_ind(d1,d2,equal_var=False)
    return t,p


def cohens_d(d1,d2):
    nx = len(d1)
    ny = len(d2)
    dof = nx + ny - 2
    if nx == ny:
        d = (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
    else:
        d = (mean(d1) - mean(d2)) / sqrt(((nx-1)*std(d1, ddof=1) ** 2 + (ny-1)*std(d2, ddof=1) ** 2) / dof)
    return d


if __name__ == "__main__":
    #%%
    params = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_stimuli\metadata/alexnet_params_20000.npy')
    label = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_face_result/activation_label_250.npy')

    label_0 = np.argwhere(label == 0).astype('int32')
    label_1 = np.argwhere(label == 1).astype('int32')

    p_sum = []
    d_sum = []

    for i in range(4092):
        i_trial = params[:, i]
        d_0 = np.squeeze(i_trial[label_0])
        d_1 = np.squeeze(i_trial[label_1])

        t, p = ttest_levene(d_0,d_1)
        d = t * sqrt(1/len(d_0) + 1/len(d_1))

        p_sum.append(p)
        d_sum.append(d)

    # find the significant parameter
    p_sum = np.array(p_sum)
    p_signIndex = np.argwhere(p_sum < (0.05/4092))
    p_minsignIndex = np.argwhere(p_sum == np.max(p_sum[p_signIndex]))

    np.save(r'D:\cnnface\Data_sorted\alexnet\param_analysis\data/d_sum.npy',d_sum)
    np.save(r'D:\cnnface\Data_sorted\alexnet\param_analysis\data/p_signIndex',p_signIndex)
    np.save(r'D:\cnnface\Data_sorted\alexnet\param_analysis\data/p_minsignIndex',p_minsignIndex)
#%%
    # d_sum_2 = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/d_sum.npy')
    # p_signIndex_2 = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/p_signIndex.npy')
    # p_minsignIndex_2 = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/p_minsignIndex.npy')
    #
    # print('d_sum is euqal:', (d_sum_2==d_sum).all())
    # print('p_signIndex is euqal:', (p_signIndex_2==p_signIndex).all())
    # print('p_minsignIndex is euqal:', (p_minsignIndex_2==p_minsignIndex).all())
    #
    #
    # print('p_sum is euqal:', (p_sum_2==p_sum).all())