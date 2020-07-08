import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnnface.analysis.generate_ci import cal_paramci,generateCI

# read male activation of alexnet and calculate the activation baseline
# noiseface_data = pd.read_csv(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_face_result/'
#                              r'250baseface_noiseface_result.csv')
# male_activation = np.array(noiseface_data['male_activation'])

# read male activation for vggface
noiseface_data = pd.read_csv(r'D:\cnnface\gender_analysis\noise_stimulus\label/noiseface_result.csv')
male_activation = np.array(noiseface_data['male_activation'])
baseline = np.mean(male_activation)

# calculate the label
activation_wave = male_activation - baseline
np.save(r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\activation_label/'
        r'vgg_activation_wave.npy',activation_wave)
label = np.array([1 if a > 0 else 0 for a in activation_wave])

np.save(r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\activation_label/'
        r'vgg_activation_label_20000.npy', label)
param_20000 = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_stimuli\metadata'
                      r'/alexnet_params_20000.npy')
param_ci = cal_paramci(param_20000, label)
scale = [2,4,8,16,32]
cis = generateCI(param_ci,level=scale)
for ci, s in zip(cis,scale):
    plt.imshow(ci,cmap='jet')
    plt.axis('off')
    plt.title(s)
    plt.savefig(r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\ci/{}.png'.format(s),dpi=300)
    plt.show()

ci = generateCI(param_ci)
plt.imshow(ci,cmap='jet')
plt.axis('off')
plt.title('CI')
plt.savefig(r'D:\cnnface\gender_analysis\supplementray_analysis\vggface_activation_label_result\ci/CI.png',dpi=300)
plt.show()

#%%
act = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\alexnet_357_sinusoid\result/act.npy')
male_activation = act[:, 1]
baseline = np.mean(male_activation)
activation_wave = male_activation - baseline
label = np.array([1 if a > 0 else 0 for a in activation_wave])
np.save(r'D:\cnnface\gender_analysis\supplementray_analysis\alexnet_357_sinusoid\result/act_label.npy',label)

old_label = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\noise_face_result/activation_label_250.npy')
print((label == old_label).all())