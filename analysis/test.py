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
def ci_show(ci,savepath=False,colorbar=True):
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


param_n = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
label = np.load(r'D:\cnnface\Data_sorted\vggface\raw/gender_label_20000.npy')

label_0 = np.argwhere(label == 0).astype('int32')
label_1 = np.argwhere(label == 1).astype('int32')

param_0 = param_n[label_0]
param_1 = param_n[label_1]

param_0 = np.squeeze(np.mean(param_0, axis=0))
param_1 = np.squeeze(np.mean(param_1, axis=0))

ci_0 = generateCI(param_0)
ci_1 = generateCI(param_1)

plt.imshow(ci_0,'gray',vmax=ci_0.max()*1.3,vmin=ci_0.min()*1.3)
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/ci0_1.3.jpg',bbox_inches='tight',pad_inches=0,dpi=300)
plt.imshow(ci_1,'gray',vmax=ci_1.max()*1.3,vmin=ci_1.min()*1.3)
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/ci1_1.3.jpg',bbox_inches='tight',pad_inches=0,dpi=300)

#%%
param_n = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
label = np.load(r'D:\cnnface\Data_sorted\vggface\raw/gender_label_20000.npy')

label_0 = np.argwhere(label == 0).astype('int32')[:3]
label_1 = np.argwhere(label == 1).astype('int32')[1:4]

param_0 = param_n[label_0]
param_1 = param_n[label_1]

for i, param in enumerate(param_0):
    param = np.squeeze(param)
    noise = generateCI(param)
    plt.imshow(noise,'gray',vmax=noise.max()*1.3,vmin=noise.min()*1.3)
    plt.axis('off')
    plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/noise_pattern\female/noise_{}.jpg'.format(i+1),bbox_inches='tight',pad_inches=0,dpi=300)


for i, param in enumerate(param_1):
    param = np.squeeze(param)
    noise = generateCI(param)
    plt.imshow(noise,'gray',vmax=noise.max()*1.3,vmin=noise.min()*1.3)
    plt.axis('off')
    plt.savefig(r'F:\研究生资料库\项目五：AI\大会报告\noisy face/noise_pattern\male/noise_{}.jpg'.format(i+1),bbox_inches='tight',pad_inches=0,dpi=300)