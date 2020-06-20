#%%
import os
import numpy as np


#%% extract the age information
pre_path = r'D:\cnnface\gender_analysis\human_result\exp\gender'
page = range(1,12,1)

# get file list
file_list = []
for p in page:
    dir_name = 'part{}'.format(p)
    file_dir =  os.path.join(pre_path,dir_name)
    file_path = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
    file_list.extend(file_path)
# open file get age
age_list = []
gender_list = []
for file in file_list:
    with open(file,'r') as fi:
        info = fi.readline().split(',')
        age = info[1]
        gender = info[2]

        age_list.append(age)
        gender_list.append(gender)

print('min age:',min(age_list),'max age:',max(age_list),'mean age:')
age_list = np.array([int(a) for a in age_list])
print('mean age:', age_list.mean())


#%%
import os
import shutil
# select the 30 id
female_list = os.listdir(r'D:\VGGface2\overlap_vggface2_female')[:15]
male_list = os.listdir(r'D:\VGGface2\overlap_vggface2_male')[:15]

id_list = female_list + male_list

oripath = r'D:\VGGface2\overlap_vggface1_2'
target_path = r'D:\AI_twostream\Data\face'

for id in id_list:
    des = os.path.join(oripath,id)
    tar = os.path.join(target_path,id)
    shutil.copytree(des,tar)

#%%
import os
import shutil
id_list = os.listdir(r'D:\AI_twostream\Data\face\train')

for id in id_list:
    oripath = os.path.join(r'D:\AI_twostream\Data\face\train',id)
    tarpath = os.path.join(r'D:\AI_twostream\Data\face\val',id)
    if os.path.exists(tarpath) is False:
        os.mkdir(tarpath)
    img_list = os.listdir(oripath)[:50]
    for img in img_list:
        img_oripath = os.path.join(r'D:\AI_twostream\Data\face\train',id,img)
        img_tarpath = os.path.join(r'D:\AI_twostream\Data\face\val',id,img)
        shutil.copy(img_oripath,img_tarpath)


#%%
from cnnface.dnn.model_reconstruct import Vgg_face
import torch

input = torch.rand(2,3,256,256)
vggface = Vgg_face()
output = vggface(input)

#%%
# generate the different frequency ci
import numpy as np
import matplotlib.pyplot as plt
from cnnface.stimuli.image_manipulate import img_similarity

cis_noise = np.load(r'D:\cnnface\Data_sorted\noise_example\data/noise_cis.npy')
ci_noise = np.load(r'D:\cnnface\Data_sorted\noise_example\data/noise_ci.npy')
saveprepath = r'D:\cnnface\Data_sorted\noise_example\img/noise_{}.jpg'
for ci,scale in zip(cis_noise,[2,4,8,16,32]):
    plt.imshow(ci,cmap='gray',vmin=ci_noise.min(),vmax=ci_noise.max())
    plt.axis('off')
    savepath = saveprepath.format(scale)
    plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)
    plt.show()

#%%
plt.imshow(ci_noise,cmap='gray',vmin=ci_noise.min(),vmax=ci_noise.max())
plt.axis('off')
savepath = saveprepath.format('all')
plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)
