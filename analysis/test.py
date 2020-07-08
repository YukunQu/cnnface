# %%
# find the 1600 images differently in two critertion

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cnnface.analysis.generate_ci import generateCI, cal_paramci
from cnnface.stimuli.image_manipulate import img_similarity

#%%
# find the difference label between critertion 1 and 2
label_c1 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
label_c2 = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\activation_label/vgg_activation_label_20000.npy')

# record: 1609 images different

# get the parameters of 1609 images
param_20000 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')

#%%

new_param_ci = cal_paramci(param_20000,label_c2)
new_ci = generateCI(new_param_ci)
new_cis = generateCI(new_param_ci, [2,4,8,16,32])

old_param_ci = cal_paramci(param_20000,label_c1)
old_ci = generateCI(old_param_ci)
old_cis = generateCI(old_param_ci, [2,4,8,16,32])
sim = img_similarity(new_ci, old_ci, 'pearsonr')
print(sim)

param_human = np.load(r'D:\cnnface\Data_sorted\human\raw/param_exp.npy')
label_human = np.load(r'D:\cnnface\Data_sorted\human\raw/label_sum.npy')
human_param_ci = cal_paramci(param_human, label_human)
human_ci = generateCI(human_param_ci)
human_cis = generateCI(human_param_ci,[2,4,8,16,32])

# different from two parameter ci
# save param_ci
np.save(r'D:\cnnface\Data_sorted\vggface\ci\data/param_ci_vgg.npy', old_param_ci)
np.save(r'D:\cnnface\Data_sorted\vggface_act\ci\data/param_ci_vgg_act.npy', new_param_ci)
np.save(r'D:\cnnface\Data_sorted\human\ci\data/param_ci_human.npy', human_param_ci)

# save CI of c1
np.save(r'D:\cnnface\Data_sorted\vggface\ci\data/ci_vgg.npy', old_ci)
np.save(r'D:\cnnface\Data_sorted\vggface\ci\data/cis_vgg.npy', old_cis)

# save CI of c2
np.save(r'D:\cnnface\Data_sorted\vggface_act\ci\data/ci_vgg_act.npy',new_ci)
np.save(r'D:\cnnface\Data_sorted\vggface_act\ci\data/cis_vgg_act.npy',new_cis)

# save CI of human
np.save(r'D:\cnnface\Data_sorted\human\ci\data/new_ci_human.npy',human_ci)
np.save(r'D:\cnnface\Data_sorted\human\ci\data/new_cis_human.npy',human_cis)


#%%
# generate the the mean of 1609 ci
diff_index = []
for i in range(20000):
    if label_c1[i] != label_c2[i]:
        diff_index.append(i)

param_diff = param_20000[diff_index]

# calculate mean and distribution of the parameters
diff_mean = np.mean(param_diff, axis=0)
sns.distplot(diff_mean)
print(diff_mean.mean())
plt.show()

#%%
old_ci = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/ci_vgg.npy')
new_ci = np.load(r'D:\cnnface\Data_sorted\vggface_act\ci\data/ci_vgg_act.npy')
def ci_show(ci, savepath=False, colorbar=True):
    plt.clf()
    plt.imshow(ci,cmap='jet',vmax=old_ci.max(),vmin=old_ci.min())
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if colorbar is True:
        plt.colorbar()
    if savepath is False:
        plt.show()
    else:
        plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)

diff_ci = generateCI(diff_mean)
ci_show(old_ci,r'F:\研究生资料库\项目五：AI\文章图\sp/old_ci_gabor.jpg')
ci_show(new_ci,r'F:\研究生资料库\项目五：AI\文章图\sp/new_ci_gabor.jpg')
#ci_show(diff_ci,r'F:\研究生资料库\项目五：AI\文章图\sp/methdo_ci_diff.jpg')

#%%
param_ci_vgg = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/param_ci_vgg.npy')
ci = generateCI(param_ci_vgg)
plt.imshow(ci,'gray')
plt.show()

param_ci_human = np.load(r'D:\cnnface\Data_sorted\human\ci\data/param_ci_human.npy')
ci = generateCI(param_ci_human)
plt.imshow(ci,'gray')
plt.show()




#%%
# generate the alexnet ci
params = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_stimuli\metadata/alexnet_params_20000.npy')
label = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_face_result/activation_label_250.npy')

alexnet_param_ci = cal_paramci(params,label)
alexnet_ci = generateCI(alexnet_param_ci)
new_cis = generateCI(alexnet_param_ci, [2,4,8,16,32])

plt.imshow(alexnet_ci,'jet')
plt.show()


#%%
param_20000 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
label = r'D:\cnnface\Data_sorted\new_alexnet/label.npy'

params_ci = cal_paramci(param_20000, label)
alexnet_ci =generateCI(params_ci)

plt.imshow(alexnet_ci,'jet')
plt.show()

#%%


def average_ci(param_n):
    ci_all = np.zeros((512, 512))
    for i, param in enumerate(param_n):
        if i%100 == 0:
            print(i)
        param = np.squeeze(param)
        ci = generateCI(param)
        ci_all = ci_all + ci
    ci_average = ci_all / len(param_n)
    return ci_average


param_n = np.load(r'D:\cnnface\Data_sorted\human\raw/param_exp.npy')
label = np.load(r'D:\cnnface\Data_sorted\human\raw/label_sum.npy')

label_0 = np.argwhere(label == 0).astype('int32')
label_1 = np.argwhere(label == 1).astype('int32')

param_0 = param_n[label_0]
param_1 = param_n[label_1]

param_0 = np.squeeze(np.mean(param_0, axis=0))
param_1 = np.squeeze(np.mean(param_1, axis=0))

ci_0 = generateCI(param_0)
ci_1 = generateCI(param_1)

plt.imshow(ci_0,'gray',vmax=ci_0.max()*1.1,vmin=ci_0.min()*1.1)
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure1\human\average/ci0_1.1.jpg',bbox_inches='tight',pad_inches=0,dpi=300)
plt.imshow(ci_1,'gray',vmax=ci_1.max()*1.1,vmin=ci_1.min()*1.1)
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure1\human\average/ci1_1.1.jpg',bbox_inches='tight',pad_inches=0,dpi=300)

#%%
param_n = np.load(r'D:\cnnface\Data_sorted\human\raw/param_exp.npy')
label = np.load(r'D:\cnnface\Data_sorted\human\raw/label_sum.npy')

label_0 = np.argwhere(label == 0).astype('int32')[:3]
label_1 = np.argwhere(label == 1).astype('int32')[:3]

param_0 = param_n[label_0]
param_1 = param_n[label_1]

for i, param in enumerate(param_0):
    param = np.squeeze(param)
    noise = generateCI(param)
    plt.imshow(noise,'gray',vmax=noise.max()*1.3,vmin=noise.min()*1.3)
    plt.axis('off')
    plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure1\human\female/femalenoise_{}.jpg'.format(i+1),bbox_inches='tight',pad_inches=0,dpi=300)


for i, param in enumerate(param_1):
    param = np.squeeze(param)
    noise = generateCI(param)
    plt.imshow(noise,'gray',vmax=noise.max()*1.3,vmin=noise.min()*1.3)
    plt.axis('off')
    plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure1\human\male/noise_{}.jpg'.format(i+1),bbox_inches='tight',pad_inches=0,dpi=300)

#%%
from cnnface.analysis.generate_ci import cal_paramci,generateCI,act2label
act = np.load(r'D:\cnnface\Data_sorted\vgg16\noiseface_label_raw/act.npy')
act_label = act2label(act, r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\activation_label/shan.npy')

# generate the vgg16 CI
param = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
#act_label = np.load(r'D:\cnnface\Data_sorted\vgg16\noiseface_label_raw/act_label.npy')

param_ci = cal_paramci(param, act_label)
ci = generateCI(param_ci)
cis = generateCI(param_ci,[2,4,8,16,32])
# np.save(r'D:\cnnface\Data_sorted\vgg16\ci\data/param_ci_lr0.01.npy', param_ci)
# np.save(r'D:\cnnface\Data_sorted\vgg16\ci\data/ci_lr0.01.npy', ci)
# np.save(r'D:\cnnface\Data_sorted\vgg16\ci\data/cis_lr0.01.npy', cis)

#%%
import  numpy as np
from PIL import Image
from  cnnface.analysis.generate_ci import recon_face

baseface = Image.open(r'D:\AI_generate_faces\baseface/frame286.png').convert('L')
ci = np.load(r'D:\cnnface\Data_sorted\alexnet\ci\data/ci_alexnet.npy')
scaleIndex = 45/ci.max()
img_add,img_sub = recon_face(baseface, ci, scaleIndex)

img_add.save(r'D:\AI_generate_faces\prototype\alexnet/female_en.jpg')
img_sub.save(r'D:\AI_generate_faces\prototype\alexnet/male_en.jpg')

#%%
import os
from cnnface.stimuli.generate_stimuli_csv import read_Imagefolder

prepath = r'D:\AI_generate_faces\stylegan_face'
picpath, condition = read_Imagefolder(prepath)
imgs_path = [os.path.join(prepath,p) for p in picpath]

img_path_female = []
img_path_male = []
for l, img_path in zip(label, imgs_path):
    if l == 0 :
        img_path_female.append(img_path)
    elif l ==1 :
        img_path_male.append(img_path)

#%%
import os
from PIL import Image

female = Image.open(r'D:\AI_generate_faces\average/female_average.png').convert('L').convert('RGB')
male = Image.open(r'D:\AI_generate_faces\average/male_average.png').convert('L').convert('RGB')

female.save(r'D:\AI_generate_faces\gray/female.png')
male.save(r'D:\AI_generate_faces\gray/male.png')

#%%
# 对比度增强
from PIL import Image
from PIL import ImageEnhance

img = Image.open(r'D:\AI_generate_faces\gray/frame280.png')
enh_con = ImageEnhance.Contrast(img)
contrast = 1.5
img_contrasted = enh_con.enhance(contrast)
img_contrasted.save(r'D:\AI_generate_faces\baseface/baseface_enhance1.3.png')


#%%
param_10000 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')[:10000]
label = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\alexnet_357_sinusoid\result/raw/act_label.npy')
param_ci = cal_paramci(param_10000, label)
ci = generateCI(param_ci)
np.save(r'D:\cnnface\gender_analysis\supplementray_analysis\alexnet_357_sinusoid\result/ci/ci.npy',ci)
plt.imshow(ci, 'jet')
plt.show()