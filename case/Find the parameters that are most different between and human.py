import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt

def paramScale(param):
    ParamScale = []
    ParamScale.append(param[:12])
    ParamScale.append(param[12:60])
    ParamScale.append(param[60:252])
    ParamScale.append(param[252:1020])
    ParamScale.append(param[1020:4092])
    return ParamScale

d_cnn = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/cohensd_unabs.npy')
d_human = np.load(r'D:\cnnface\gender_analysis\human_result\para_significant/cohensd_unabs.npy')

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

sns.set_style('darkgrid')
sns.swarmplot(scale, diffValue)
sns.lineplot(x=bin,y=y)
sns.lineplot(x=bin,y=-y)
plt.show()

np.save(r'D:\cnnface\gender_analysis\Result\difference/diffValue.npy',diffValue)
diffValueAbs = np.abs(diffValue)
highDiffIndex = np.argwhere(diffValueAbs > 2.49)