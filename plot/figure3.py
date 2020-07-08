
# paper figure 3: Mahatoon map

# Plot Manhattan map of distance which measure the distance of two distributions

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
sns.set_context('paper')

# prepare data
# distance = np.load(r'D:\cnnface\Data_sorted\human\param_analysis\data/d_sum.npy')
# p_minsignIndex = np.load(r'D:\cnnface\Data_sorted\human\param_analysis\data/p_minsignIndex.npy')

distance = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\result\p_sign/cohensd_sum.npy')
p_minsignIndex = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\result\p_sign/p_minsignIndex.npy')
distance = np.abs(distance)

scale2 = ['scale2'] * 12
scale4 = ['scale4'] * 48
scale8 = ['scale8'] * 192
scale16 = ['scale16'] * 768
scale32 = ['scale32'] * 3072
scale = scale2 + scale4 + scale8 + scale16 + scale32

plt.figure(figsize =(6,6))

# plot the manhattan map
#sns.set_palette(sns.cubehelix_palette(6)[-5:])
#sns.set_palette(sns.color_palette("Blues"))
#sns.set_palette(sns.color_palette("coolwarm", 7))
#sns.set_palette(sns.light_palette((210, 90, 60), input="husl"))
#sns.set_palette(sns.color_palette(['#1f77b4','#FF9478','#40b1c9','#d62728','#9467bd']))

sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))


snsplt = sns.swarmplot(x=scale, y=distance)
plt.tick_params(labelsize=14)
plt.xticks([0,1,2,3,4,5],['2','4','8','16','32'], size=14)
plt.yticks(size=14)
plt.ylabel("|Cohen's d|",size=16)
plt.xlabel('Scale (cycles/image)',size=16)
# plot the significant line
bin = np.arange(-0.2,4.3,0.2)
y = np.full((len(bin)),distance[p_minsignIndex])
snsplt = sns.lineplot(x=bin,y=y,color='#35476D')

# save figure
fig = snsplt.get_figure()
fig.savefig(r'D:\cnnface\Data_sorted\alexnet\param_analysis\img/alexnet_dManhatann_old.jpg',dpi=300,bbox_inches='tight')
plt.show()

#%%
correlation = np.load(r'D:\cnnface\Data_sorted\compareResult\contribution\new/high_info_parameter_correlation.npy')
sns.set_context('paper')
plt.figure(figsize =(6,6))
sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
sns.barplot([2,4,8,16,32],correlation)
plt.tick_params(labelsize=14)
plt.ylabel('Correlation',size=16)
plt.xlabel('Scale(cycles/image)',size=16)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\new/high_info_parameter_correlation.jpg',dpi= 300,
            bbox_inches='tight')
plt.show()