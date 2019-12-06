# 将cohens‘d 的d值每个scales 对应位置的权重，生成对应的权重heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load conhens’d of all parameter
conhensd = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/cohensd_sum.npy')

# load patchesIdx, patchesIdx contains the position which each parameter occupy.
patchIdx = np.load(r'D:\cnnface/patchidx.npy').astype('int32')
paramWeight = conhensd[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)

weightMap_sum = np.sum(paramWeight, axis=2)

weightMap_scales = []
slice_index = {2: 0, 4: 12, 8: 24, 16: 32, 32: 48}
for i in [2,4,8,16,32]:
    index = slice_index[i]
    weightMap_scale = np.sum(paramWeight[:, :, index:index + 12],axis=2)
    weightMap_scales.append(weightMap_scale)
weightMap_scales = np.array(weightMap_scales)

plt.imshow(weightMap_sum)
plt.axis('off')
plt.colorbar()
plt.show()

for wm in weightMap_scales:
    plt.imshow(wm)
    plt.axis('off')
    plt.colorbar()
    plt.show()