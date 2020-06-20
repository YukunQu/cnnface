import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cnnface.analysis.parameter_analysis import index2diffscale

def dice(a1,a2):
    intersect = np.intersect1d(a1,a2)
    son = 2 *len(intersect)
    mother = len(a1) + len(a2)
    diceIndex = son/mother
    return diceIndex


paraIndexOver80_human = np.load(r'D:\cnnface\gender_analysis\Result\difference/paraIndexOver80_human.npy')
paraIndexOver80_cnn = np.load(r'D:\cnnface\Data_sorted\compareResult\contribution/d_cnn_ContriOver80Index.npy')

index_vgg = index2diffscale(paraIndexOver80_cnn)
index_human = index2diffscale(paraIndexOver80_human)

dices = []
for v,h in zip(index_vgg,index_human):
    di = dice(v,h)
    dices.append(di)

bins = [2,4,8,16,32]
sns.set_style('darkgrid')
sns.barplot(bins,dices)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\contribution/dice.png',dpi=300)
plt.show()

np.save(r'D:\cnnface\Data_sorted\compareResult\contribution/dice.npy',dices)
