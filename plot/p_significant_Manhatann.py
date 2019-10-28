import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

# prepare data
p_signIndex = np.load(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/p_signIndex.npy')
distance = np.load(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/s_sum.npy')
distance = np.abs(distance)
p_minsignIndex = np.load(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/p_minsignIndex.npy')

scale2 = ['scale2'] * 12
scale4 = ['scale4'] * 48
scale8 = ['scale8'] * 192
scale16 = ['scale16'] * 768
scale32 = ['scale32'] * 3072
scale = scale2 + scale4 + scale8 + scale16 + scale32

hue = []
for i in range(4092):
    if i in p_signIndex:
        hue.append('Significant')
    else:
        hue.append('Unsignificant')

# plot the manhattan map
snsplt =sns.swarmplot(x=scale,y=distance)
plt.tick_params(labelsize=12)
# plot the significant line

bin = np.arange(0,5)
y = np.full((5,),distance[p_minsignIndex])
snsplt = sns.lineplot(x=bin,y=y)

fig = snsplt.get_figure()
fig.savefig(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/Manhatannx.jpg')
plt.show()

# proportion bar plot
scale2_pSign = p_signIndex[p_signIndex<12 ]
scale4_pSign = p_signIndex[(p_signIndex>=12) & (p_signIndex<60)]
scale8_pSign = p_signIndex[(p_signIndex>=60) & (p_signIndex<252)]
scale16_pSign = p_signIndex[(p_signIndex>=252) & (p_signIndex<1020)]
scale32_pSign = p_signIndex[(p_signIndex>=1020) & (p_signIndex<4092)]

pSign_prop = [len(scale2_pSign)/12,len(scale4_pSign)/48,len(scale8_pSign)/192,
              len(scale16_pSign)/768,len(scale32_pSign)/3072]
bins = ['scale2','scale4','scale8', 'scale16', 'scale32']

sns.barplot(bins,pSign_prop)
plt.savefig(r'D:\cnnface\Emotion_analysis\CI_analysis\para_significant/neutral/pSing_prop.jpg')