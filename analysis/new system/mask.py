import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as ns

#%%
ci = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask/ci_cnn.npy')
ci_img = Image.fromarray(ci).resize((224, 224))
ci = np.array(ci_img)
ci_mask1 = np.zeros_like(ci)
ci_mask2 = np.zeros_like(ci)
ci_mask1[ci > 0] = ci[ci > 0]
ci_mask2[ci < 0] = ci[ci < 0]

plt.imshow(ci_mask1, cmap='jet', vmin=ci.min(), vmax=ci.max())
plt.show()

savepath = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask\mask-2'
np.save(savepath + '/ci_mask1.npy', ci_mask1)
np.save(savepath + '/ci_mask2.npy', ci_mask2)

masks = np.array([ci_mask1, ci_mask2])
np.save(savepath + '/masks.npy', masks)


#%%
def generate_mosaic_mask(ci, mosaic_num, savepath, scale=True, shuffle=False):
    # How many mosaics does an edge contain
    if scale:
        ci_img = Image.fromarray(ci).resize((224, 224))
        ci = np.array(ci_img)
    if ci.shape[0] == ci.shape[1]:
        dim = ci.shape[0]
    else:
        print('The ci must be a square.')
    mosaic_size = int(dim/mosaic_num)

    if shuffle:
        ci = ci.reshape(-1)
        ns.shuffle(ci)
        ci = ci.reshape((224, 224))

    masks = []
    for y_step in range(mosaic_num):
        for x_step in range(mosaic_num):
            mosaic = np.zeros_like(ci)
            mosaic[y_step*mosaic_size:(y_step+1)*mosaic_size, x_step*mosaic_size:(x_step+1)*mosaic_size] = \
                ci[y_step*mosaic_size:(y_step+1)*mosaic_size, x_step*mosaic_size:(x_step+1)*mosaic_size]
            masks.append(mosaic)
            plt.imshow(mosaic, 'jet', vmax=ci.max(), vmin=ci.min())
            plt.savefig(savepath+'/{}_{}.jpg'.format(x_step, y_step))
            plt.clf()
    masks = np.array(masks)
    print(masks.shape)
    np.save(savepath+'/masks.npy', masks)

ci = np.load(r'D:\cnnface\gender_analysis\CI_analysis/ci_cnn.npy')
savepath = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask\random-mask-64'
mosaic_num = 8

generate_mosaic_mask(ci, mosaic_num, savepath, shuffle=True)
