# ------------------------------------------------------------------------------ #
# generate the Single parameter face image                                       #
# ------------------------------------------------------------------------------ #

import numpy as np
from PIL import Image
import time
import subprocess
import os
from cnnface.analysis.generate_ci import generateCI,recon_face

nor = lambda x: (x - x.min()) / (x.max() - x.min())
baseface = np.array(Image.open(r'D:\cnnface\gender_analysis\face_template\gray/baseface.jpg')).astype('int64')
p_signIndex = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/p_signIndex.npy')
param_ci = np.load(r'D:\cnnface\gender_analysis\CI_analysis/param_ci_20000.npy')
patches = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patches.npy')
patchIdx = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/patchidx.npy').astype('int64')


# Generate the 244 face images which contains baseFace and add single parameter noise
start = time.time()
for index in p_signIndex:
    param = np.zeros(4092)
    for th,step in enumerate(np.arange(-1, 1, 2/20)):
        param[index] = step
        patchParam = param[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)
        noise = np.sum(patches * patchParam, axis=2)
        noise = noise * 100
        noiseface = baseface + noise
        noiseface_img = Image.fromarray(noiseface).convert('RGB')
        if os.path.exists('D:\cnnface\gender_analysis\CI_analysis\param_effect\single_param_img/{}'.format(index+1))==False:
            os.mkdir('D:\cnnface\gender_analysis\CI_analysis\param_effect\single_param_img/{}'.format(index+1))
        noiseface_img.save('D:\cnnface\gender_analysis\CI_analysis\param_effect\single_param_img'
                           '/{}/param{}_{}th.jpg'.format(index+1,index+1, th+1))
        midtime = time.time()
        cost_time = time.time() - midtime
        print('the {} image of parameter {}  have been saved.'.format(th+1, index+1))
        print('Testing complete in {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))

end = time.time()
cost_time = end - start
print('Testing complete in {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))