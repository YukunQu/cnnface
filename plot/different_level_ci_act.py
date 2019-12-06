# '将ci分成五个空间频率水平加到base face上，检测网络对五个水平的判断概率'

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

act = np.load(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/cnn_reconface_act.npy')
label_prob = np.load(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/cnn_reconface_prob.npy')

act_add = act[:5][:,0]
act_sub = act[5:10][:,1]

label_prob_add = label_prob[:5][:,0]
label_prob_sub = label_prob[5:10][:,1]


category = [2,4,8,16,32] * 2
y = np.hstack((act_add,act_sub))
hue = ['female']*5 +['male']*5
sns.barplot(x=category, y= y,hue=hue)
plt.show()

category = [2,4,8,16,32]
sns.lineplot(x=category, y= label_prob_add)
sns.lineplot(x=category, y= label_prob_sub)
plt.xticks([2,4,8,16,32])
plt.xlabel('Scale',size=16)
plt.ylabel('Probability',size=16)
plt.show()