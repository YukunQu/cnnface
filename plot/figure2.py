import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from cnnface.stimuli.image_manipulate import nor
from cnnface.analysis.ci_analysis import correlation_ci


cis_human = np.load(r'D:\cnnface\gender_analysis\human_result\CIs/cis_human.npy')
cis_vgg = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/cis_vgg.npy')

cis = np.concatenate((cis_vgg,cis_human),axis=0)
fig = plt.figure(figsize=(26, 9),dpi=300)
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 5),
                axes_pad=1.4,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.5
                )

for ax,ci in zip(grid,cis):
    ax.set_axis_off()
    ci = nor(ci)
    im = ax.imshow(ci,cmap='jet')

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'], fontsize=28)
cbar.ax.set_yticks(np.arange(0, 1.1, 0.2))
plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure2/figure2.jpg',dpi=300)

similarity = correlation_ci(cis_vgg,cis_human,True)
print(similarity)

for k,v in simi.items():
    simi[k] = np.round(v[0],2)