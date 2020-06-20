import numpy as np
import matplotlib.pyplot as plt
from cnnface.analysis.generate_ci import generateCI
from cnnface.stimuli.image_manipulate import nor
from mpl_toolkits.axes_grid1 import AxesGrid

param_1 = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')[0]

noise_ci = generateCI(param_1)
noise_cis = generateCI(param_1, [2,4,8,16,32])
noise_cis = [noise for noise in noise_cis]
#
# np.save(r'F:\研究生资料库\项目五：AI\文章图\sp\noise_ci.npy', noise_ci)
# np.save(r'F:\研究生资料库\项目五：AI\文章图\sp\noise_cis.npy', noise_cis)

plt.imshow(noise_ci, 'gray')
plt.axis('off')
plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\sp/noise_ci_new.jpg',bbox_inches='tight',pad_inches=0,dpi=300)
plt.show()

fig = plt.figure(figsize=(18, 3),dpi=300)
grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 5),
                axes_pad=1.0,
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.5
                )

for ax,ci in zip(grid,noise_cis):
    im = ax.imshow(ci,'gray', vmin=noise_ci.min(), vmax=noise_ci.max())
    ax.axis('off')

plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\sp/noise_cis_new.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()

# generate the sinusoid noise
patches = np.load(r'D:\cnnface\Data_sorted\commonData/patches.npy')
patchIdx = np.load(r'D:\cnnface/patchidx.npy').astype('int32')
patchParam = param_1[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)
gabors = patches[:, :, 0:12] * patchParam[:, :, 0:12]


fig = plt.figure(figsize=(25, 9),dpi=300)
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 6),
                axes_pad=1.0,
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.5
                )

for i,ax in enumerate(grid):
    gabor = gabors[:,:,i]
    #im = ax.imshow(gabor, 'gray',vmin=noise_cis[0].min(), vmax=noise_cis[0].max())
    im = ax.imshow(gabor, 'gray')
    ax.axis('off')
    ax.set_title(str(np.round(param_1[i],2)), y=-0.15, fontsize=16)

plt.savefig(r'F:\研究生资料库\项目五：AI\文章图\sp/gabors_new.jpg', dpi=300,bbox_inches='tight',pad_inches=0)
