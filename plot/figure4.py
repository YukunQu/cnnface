import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from cnnface.stimuli.image_manipulate import nor
from cnnface.plot.figure1 import ci_show


cis_alexnet = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\ci_result/cis_alexnet.npy')

fig = plt.figure(figsize=(26, 5),dpi=300)
grid = AxesGrid(fig,111,
                nrows_ncols=(1, 6),
                axes_pad=0.5,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2
                )
scales = ['2','4','8','16','32']
for ax,ci,scale in zip(grid,cis_alexnet,scales):
    ax.set_axis_off()
    ci = nor(ci)

    im = ax.imshow(ci,cmap='jet')
    ax.set_title(scale,y = -0.2,size=28)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticklabels(['0', '0.5','1'], fontsize=28)
cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\img\Figure4/figure4.jpg',dpi=300)
plt.show()

#%%
ci_alexnet = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\ci_result/ci_alexnet.npy')
save_path = r'D:\cnnface\Data_sorted\alexnet\ci/ci_alexnet.jpg'
ci_show(ci_alexnet,save_path,True)

