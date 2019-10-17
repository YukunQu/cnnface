# ------------------------------------------------------------------------------ #
# generate the Single parameter face image                                       #
# ------------------------------------------------------------------------------ #

import numpy as np
from PIL import Image
import time
import subprocess

nor = lambda x: (x - x.min()) / (x.max() - x.min())
baseface = np.array(Image.open(r'/nfs/h1/workingshop/quyukun/DNN/result_data/face_template/frame054_gray_512.jpg')).astype('int64')
p_signIndex = np.load(r'/nfs/h1/workingshop/quyukun/DNN/result_data/ci/p_signIndex.npy')
patches = np.load(r'/nfs/h1/workingshop/quyukun/DNN/result_data/ci/patches.npy')
patchIdx = np.load('/nfs/h1/workingshop/quyukun/DNN/result_data/ci/patchidx.npy').astype('int64')

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
        subprocess.call('mkdir {}'.format(index+1))
        noiseface_img.save('/nfs/h1/workingshop/quyukun/DNN/result_data/'
                           'SingleParameter_image/{}/param{}_{}th.jpg'.format(index,index+1, th+1))
        midtime = time.time()
        cost_time = time.time() - midtime
        print('the {} image of parameter {}  have been saved.'.format(th+1, index+1))
        print('Testing complete in {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))

end = time.time()
cost_time = end -start
print('Testing complete in {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))

