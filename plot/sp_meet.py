import os
import numpy as np
import pandas as pd
from PIL import Image
from cnnface.stimuli.image_manipulate import nor


# calculate the average face

# load label and get Index
label = np.load(r'D:\cnnface\Data_sorted\vggface\raw/gender_label_20000.npy')
label_0 = np.argwhere(label == 0).astype('int32')
label_1 = np.argwhere(label == 1).astype('int32')

# read csv file
csv_file_path = r'D:\cnnface\gender_analysis\noise_stimulus/baseface_20000.csv'
with open(csv_file_path,'r') as f:
    picpath = f.readline().rstrip()
csv_file = pd.read_csv(csv_file_path, skiprows=1)
picname = np.array(csv_file['stimID'])
condition = np.array(csv_file['condition'])
picname0 = picname[label_0]
picname1 = picname[label_1]

# get images and average :
result_images = np.zeros((512,512), np.float32)
num_images = len(label_1)
for idx,picname_idx in zip(range(num_images),picname1):
    if idx % 100 == 0:
        print(idx)
    picimg = Image.open(os.path.join(picpath, picname_idx[0]))
    picimg = np.array(picimg).astype(np.float32)
    if idx == 0:
        print(picimg.shape)
    result_images = result_images + picimg

result_image = np.uint8(result_images / num_images)
result_img = Image.fromarray(result_image)
result_img.save(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/average_face_1.jpg')

#%%

img_0 = np.array(Image.open(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/average_face_0.jpg')).astype('float32')
img_1 = np.array(Image.open(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/average_face_1.jpg')).astype('float32')

ci = img_0 - img_1

plt.imshow(ci*10,cmap='jet')
plt.show()

#%%
import os
import numpy as np
from PIL import Image
from cnnface.analysis.generate_ci import generateCI
from cnnface.analysis.prototype_analysis import PrototypeFace
param_n = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
label = np.load(r'D:\cnnface\Data_sorted\vggface\raw/gender_label_20000.npy')
label_0 = np.argwhere(label == 0).astype('int32')
label_1 = np.argwhere(label == 1).astype('int32')

param_0 = param_n[label_0]
param_1 = param_n[label_1]

# average the parameters after labeling
param_0 = np.squeeze(np.mean(param_0, axis=0))
param_1 = np.squeeze(np.mean(param_1, axis=0))

ci_0 = generateCI(param_0)
ci_1 = generateCI(param_1)

baseface = Image.open(r'D:\cnnface\Data_sorted\commonData/baseface.jpg')

vgg_prototype = PrototypeFace(baseface,ci_0)
scaleIndex = 45/ci_0.max()
face_0,_ = vgg_prototype.recon_face(scale=scaleIndex)
vgg_prototype = PrototypeFace(baseface,ci_1)
scaleIndex = 45/ci_1.max()
face_1,_ = vgg_prototype.recon_face(scale=scaleIndex)


face_0.save(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/face_0.jpg')
face_1.save(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/face_1.jpg')

#%%
import matplotlib.pyplot as plt

plt.imshow(ci_0,'jet')
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/ci_0_jet.jpg',bbox_inches='tight', pad_inches=0,dpi=300)
plt.imshow(ci_1,'jet')
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/ci_1_jet.jpg',bbox_inches='tight', pad_inches=0,dpi=300)