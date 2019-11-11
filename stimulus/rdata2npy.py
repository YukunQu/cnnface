# load RData file to python
import rpy2.robjects as robjects
import numpy as np

robjects.r['load']("D:/cnnface/female_male_test_51_addnoise/Face_template/"
                   "meta_data/pythonRead/sinusoid1000_patches.RData")

x = robjects.r['patches']
print(np.array(x))

#%%

# convert .RData to npy file
import rpy2.robjects as robjects
import numpy as np


robjects.r['load'](r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part1.Rdata")
params1000 = np.array(robjects.r['stimuli_params'][0])

robjects.r['load'](r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part2.Rdata")
params2000 = np.array(robjects.r['stimuli_params'][0])

robjects.r['load'](r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part3.Rdata")
params3000 = np.array(robjects.r['stimuli_params'][0])

robjects.r['load'](r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part4.Rdata")
params4000 = np.array(robjects.r['stimuli_params'][0])

robjects.r['load'](r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part5.Rdata")
params5000 = np.array(robjects.r['stimuli_params'][0])

params_5000 = np.concatenate((params1000, params2000, params3000, params4000, params5000), axis=0)

np.save(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_5000', params_5000)


#%%
# convert patches and patchIdx to npy file
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