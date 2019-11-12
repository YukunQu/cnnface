# 选分类概率大于40%的那几个参数，重建CI图像，没必要转化成Image，直接heatmap。

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cnnface.analysis.generate_ci import generateCI

p_signIndex = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/p_signIndex.npy')
dist_param = np.load(r'D:\cnnface\female_male_test_51_addnoise\frame054/dist_params.npy')

# highEffectPara = np.argwhere(dist_param >= 0.4)
# highEffectParaIndex = p_signIndex[highEffectPara]

patches = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patches.npy')
patchIdx = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patchidx.npy').astype('int64')

params_ci = np.load('D:\cnnface\gender_analysis\CI_analysis/param_ci.npy')
param = np.zeros(4092)
for i in p_signIndex:
    param[i] = params_ci[i]

ci = generateCI(param)

plt.imshow(ci*90,cmap='jet')
plt.show()

# %%

plt.imshow(ci)

# %%

from PIL import Image
from cnnface.stimulus.Image_process import nor

ci_nor = nor(ci) * 255

ci_img = Image.fromarray(ci_nor)
ci_img.show()


sns.set_style('darkgrid')
sns.distplot(ci.reshape(-1))

#%%

ci_mask40 = np.zeros(ci.shape)
ci_mask40[np.abs(ci)>0.45] = 1
ci_mask40 = ci_mask40.astype('bool')

#%%

baseface = Image.open(r'D:\cnnface\female_male_test_51_addnoise\frame054/frame054_gray_512.jpg')
baseface_arr = np.array(baseface)
baseface_40_arr = np.zeros(ci.shape)
baseface_40_arr[ci_mask40] = baseface_arr[ci_mask40]
baseface_40 = Image.fromarray(baseface_40_arr).convert('RGB')
baseface_40.save(r'D:\cnnface\female_male_test_51_addnoise\Face_template\classification_noise/baseface_45.jpg')

#%%

plt.imshow(baseface_40_arr)