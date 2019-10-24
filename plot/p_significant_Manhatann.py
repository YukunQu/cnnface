import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

# prepare data
p_signIndex = np.load('D:\cnnface/female_male_test_51_addnoise\Face_template\CI_analysis/p_signIndex.npy')
distance = np.load('D:\cnnface/female_male_test_51_addnoise\Face_template\CI_analysis/s_sum.npy')
distance = np.abs(distance)

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
y = np.full((5,),distance[2216])
snsplt = sns.lineplot(x=bin,y=y)

fig =snsplt.get_figure()
fig.savefig('D:\cnnface/female_male_test_51_addnoise\Face_template\CI_analysis/Manhatannx.jpg')
plt.show()