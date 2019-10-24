import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# generate the effective parameter array
# load p_sign_parameter index and parameter effect
parameter = np.zeros(4092)
p_signIndex = np.load('D:\cnnface/female_male_test_51_addnoise\Face_template\CI_analysis/p_signIndex.npy')
p_sign_effect = np.load(r'D:\cnnface\female_male_test_51_addnoise\frame054/dist_params.npy')
for index,effect in zip(p_signIndex,p_sign_effect):
    parameter[index] = effect

# generate the CI with patches,patchIdx,parameters
patches = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patches.npy')
patchIdx = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patchidx.npy').astype('int64')

patchParam = parameter[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)
noise = np.sum(patches * patchParam, axis=2)
noise = noise * 100
noise = (noise - noise.min()) / (noise.max() - noise.min())

# plot heat map
ci_map = plt.imshow(noise,cmap = 'jet',interpolation='nearest')