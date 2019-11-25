import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

p_signIndex = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/p_signIndex.npy')
dist_param = np.load(r'D:\cnnface\gender_analysis\CI_analysis\param_effect/dist_params.npy')

scale = np.arange(len(p_signIndex))
scale[p_signIndex<12] = 2
scale[(p_signIndex>=12) & (p_signIndex<60)] = 4
scale[(p_signIndex>=60) & (p_signIndex<252)] = 8
scale[(p_signIndex>=252) & (p_signIndex<1020)] = 16
scale[(p_signIndex>=1020) & (p_signIndex<4092)] = 32

sns.swarmplot(x=scale, y=dist_param)
plt.tick_params(labelsize=14)
plt.show()