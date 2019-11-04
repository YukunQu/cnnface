# load RData file to python
import rpy2.robjects as robjects
import numpy as np

robjects.r['load'](
    "D:/cnnface/female_male_test_51_addnoise/Face_template/meta_data/pythonRead/sinusoid1000_patches.RData")

x = robjects.r['patches']
print(np.array(x))

# %%

# convert .RData to npy file
import rpy2.robjects as robjects
import numpy as np

robjects.r['load'](r"D:\cnnface\Emotion_analysis\noise_metadata/neu_run1.Rdata")
params = robjects.r['stimuli_params'][0]

# robjects.r['load'](r"D:\cnnface\Identity_analysis\meat_data/run2.Rdata")
# params = robjects.r['stimuli_params'][0]

params_5000_ori = np.array(params)
params_5000_inv = -np.array(params)
params_5000 = np.zeros((5000, 4092))
for i in range(2500):
    params_5000[i * 2 + 1, :] = params_5000_ori[i, :]
    params_5000[i * 2, :] = params_5000_inv[i, :]

# p = np.concatenate((params_1000,params_5000),axis=0)

np.save(r'D:\cnnface\Emotion_analysis\noise_metadata/neu_params_5000', params_5000)


#%%
# convert .RData to npy file
import rpy2.robjects as robjects
import numpy as np

robjects.r['load'](
    "D:/cnnface/female_male_test_51_addnoise/Face_template/meta_data/rcic_seed_1_time_9月_18_2019_10_07.Rdata")
p = robjects.r['p']
print(p.names)
patches = p[0]
patchidx = p[1]

patches = np.array(patches)
patchidx = np.array(patchidx)

np.save('D:/cnnface/female_male_test_51_addnoise/Face_template/meta_data/patches.npy', patches)
np.save('D:/cnnface/female_male_test_51_addnoise/Face_template/meta_data/patchidx.npy', patchidx)