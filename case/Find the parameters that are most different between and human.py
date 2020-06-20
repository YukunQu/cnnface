import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt
from cnnface.analysis.parameter_analysis import index2diffscale

def paramScale(param):
    ParamScale = []
    ParamScale.append(param[:12])
    ParamScale.append(param[12:60])
    ParamScale.append(param[60:252])
    ParamScale.append(param[252:1020])
    ParamScale.append(param[1020:4092])
    return ParamScale

d_cnn = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/d_sum.npy')
d_human = np.load(r'D:\cnnface\Data_sorted\human\param_analysis\data/d_sum.npy')

d_cnnScaled = scale(d_cnn)
d_humanScaled = scale(d_human)

diffValue = d_cnnScaled - d_humanScaled

diffstd = diffValue.std()
thr = 1.96 * diffstd

scale2 = ['scale2'] * 12
scale4 = ['scale4'] * 48
scale8 = ['scale8'] * 192
scale16 = ['scale16'] * 768
scale32 = ['scale32'] * 3072
scale = scale2 + scale4 + scale8 + scale16 + scale32

bin = np.arange(0,6)
y = np.full((6,), thr)

sns.set_context('paper')
sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
sns.swarmplot(scale, diffValue)
sns.lineplot(x=bin,y=y,color='#35476D')
sns.lineplot(x=bin,y=-y,color='#35476D')
plt.yticks(size=12)
plt.xticks(size=12)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\img/highDiff.png',dpi=300)
plt.show()

np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\data/diffValue.npy',diffValue)

#%%
# number of different scale
diffValueAbs = np.abs(diffValue)
highDiffIndex = np.argwhere(diffValueAbs >= thr)
highdiffScaled = index2diffscale(highDiffIndex)
highDiff_num = [len(a) for a in highdiffScaled]

np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\data/highDiff_num',highDiff_num)

sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
sns.barplot(x=[2,4,8,16,32],y=highDiff_num)
plt.yticks(size=12)
plt.xticks(size=12)
plt.ylabel('Number',size=14)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\img/highDiff_num.jpg',dpi=300)
plt.show()