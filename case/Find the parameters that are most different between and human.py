import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt
#%%

def index2diffscale(index):
    indexSubScale = []
    indexSubScale.append(index[index<12])
    indexSubScale.append(index[(index>=12) & (index<60)])
    indexSubScale.append(index[(index>=60) & (index<252)])
    indexSubScale.append(index[(index>=252) & (index<1020)])
    indexSubScale.append(index[(index>=1020) & (index<4092)])
    return indexSubScale


def paramScale(param):
    ParamScale = []
    ParamScale.append(param[:12])
    ParamScale.append(param[12:60])
    ParamScale.append(param[60:252])
    ParamScale.append(param[252:1020])
    ParamScale.append(param[1020:4092])
    return ParamScale

d_cnn = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/d_sum.npy')
d_human = np.load(r'D:\cnnface\Data_sorted\human\param_analysis\data/d_sum.npy')

d_cnnScaled = scale(d_cnn)
d_humanScaled = scale(d_human)

diffValue = d_cnnScaled - d_humanScaled

diffstd = diffValue.std()
thr = 1.96 * diffstd
#
# scale2 = ['scale2'] * 12
# scale4 = ['scale4'] * 48
# scale8 = ['scale8'] * 192
# scale16 = ['scale16'] * 768
# scale32 = ['scale32'] * 3072
# scale = scale2 + scale4 + scale8 + scale16 + scale32
#
# bin = [0,0.8,2,3,4,4.5]
# y = np.full((6,), thr)
#
# sns.set_context('paper')
# sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
# sns.swarmplot(scale, diffValue)
# sns.lineplot(x=bin,y=y,color='#35476D')
# sns.lineplot(x=bin,y=-y,color='#35476D')
# plt.xticks([0,1,2,3,4,5],['2','4','8','16','32'], size=14)
# plt.yticks(np.arange(-10,11,2),size=14)
#
# plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\img/highDiff_new.png',dpi=300)
# plt.show()
#
# np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\data/diffValue.npy',diffValue)


#%%

# number of different scale
diffmean = np.mean(diffValue)
diffValueAbs = np.abs(diffmean+diffValue)

highDiffIndex = np.argwhere(diffValueAbs >= thr)
highdiffScaled = index2diffscale(highDiffIndex)
highDiff_num = [len(a) for a in highdiffScaled]

np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\data/highDiff_num',highDiff_num)

sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
sns.barplot(x=[2,4,8,16,32],y=highDiff_num)
plt.yticks(size=14)
plt.xticks(size=14)

plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\img/highDiff_num.jpg',dpi=300)
plt.show()