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


for i in range(1,21):
    rdata = r"D:\cnnface\gender_analysis\noise_stimulus\metadata/part{}.Rdata".format(i)
    robjects.r['load'](rdata)
    params_part = np.array(robjects.r['stimuli_params'][0])
    if i == 1:
        params_20000 = params_part
    else:
        params_20000 = np.concatenate((params_20000, params_part), axis=0)

np.save(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000', params_20000)

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