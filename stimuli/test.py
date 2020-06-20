import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cnnface.stimuli.image_manipulate import img_similarity,nor
#%%
patches_old = np.load(r'D:\cnnface\Data_sorted\commonData/gabor_patches.npy')
patchIdx_old = np.load(r'D:\cnnface/patchidx.npy').astype('int32')

patches_gabor = np.load(r'D:\cnnface\patches.npy')
patchIdx_gabor = np.load(r'D:\cnnface\Data_sorted\commonData/x_patchidx.npy').astype('int32')

print((patches_old == patches_gabor).all())
print((patchIdx_old == patchIdx_gabor).all())


#%%
from cnnface.analysis.generate_ci import generateCI

param_1 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')[0]
ci = generateCI(param_1)

baseface = Image.open(r'D:\cnnface\gender_analysis\face_template\gray/baseface.jpg')
print(np.array(baseface).shape)
baseface = (np.array(baseface).astype('float64'))
noiseface = ci * 8 + baseface

noiseface_rcicr = Image.open(r'D:\cnnface\gender_analysis\noise_stimulus\baseface\part001/rcic_base_face_1_00001_ori.png')
noiseface_rcicr = np.array(noiseface_rcicr).astype('float64')

similarity = img_similarity(noiseface_rcicr,noiseface,'pearsonr')
print(similarity)

#%%

plt.imshow(noiseface,'gray')
plt.show()

#%%

prepath = r'D:\cnnface\gender_analysis\noise_stimulus\baseface\part002'
part1 = os.listdir(prepath)
face_name = part1[80]
face_path = os.path.join(prepath,face_name)
face = np.array(Image.open(face_path))

face_path2 = r'D:\cnnface\val_analysis\noiseface\part2/rcic_baseface_40921001_00081_ori.png'

face_sinu = np.array(Image.open(face_path2))

similarity = img_similarity(face, face_sinu, 'pearsonr')
print(similarity)