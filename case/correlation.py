import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

d_vgg = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/d_sum.npy')
d_human = np.load(r'D:\cnnface\Data_sorted\human\param_analysis\data/d_sum.npy')

indexOver80 = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/d_vgg_ContriOver80Index.npy')

def data2diffscale(data,index=None):
    dataSubScale = []
    if type(index) is np.ndarray:
        dataSubScale.append(data[index[index<12]])
        dataSubScale.append(data[index[(index>=12) & (index<60)]])
        dataSubScale.append(data[index[(index>=60) & (index<252)]])
        dataSubScale.append(data[index[(index>=252) & (index<1020)]])
        dataSubScale.append(data[index[(index>=1020) & (index<4092)]])
    else:
        dataSubScale.append(data[:12])
        dataSubScale.append(data[12:60])
        dataSubScale.append(data[60:252])
        dataSubScale.append(data[252:1020])
        dataSubScale.append(data[1020:])

    return dataSubScale

def index2diffscale(index):
    indexSubScale = []
    indexSubScale.append(index[index<12])
    indexSubScale.append(index[(index>=12) & (index<60)])
    indexSubScale.append(index[(index>=60) & (index<252)])
    indexSubScale.append(index[(index>=252) & (index<1020)])
    indexSubScale.append(index[(index>=1020) & (index<4092)])
    return indexSubScale

cnnContriFreq = data2diffscale(d_vgg, indexOver80)
humanContriFreq = data2diffscale(d_human,indexOver80)
res = [spearmanr(cnn,human) for cnn, human in zip(cnnContriFreq,humanContriFreq)]
correlation = [r[0] for r in res]
p = [r[1] for r in res]
p = [spearmanr(cnn,human)[1] for cnn, human in zip(cnnContriFreq,humanContriFreq)]


#%% plot
sns.set_context('paper')
plt.figure(figsize =(6,6))
sns.set_palette(sns.color_palette(['#FFBEA9','#FF9478','#C9573D','#673029','#35476D']))
sns.barplot([2,4,8,16,32],correlation)
plt.ylabel('Correlation',size=16)
#plt.xlabel('Scale(cycles/image)',size=16)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution\new/high_info_parameter_correlation.jpg',dpi= 300)
plt.show()

np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\new/high_info_parameter_correlation.npy',correlation)
np.save(r'D:\cnnface\Data_sorted\compareResult\contribution\new/correlation_significance.npy',p)