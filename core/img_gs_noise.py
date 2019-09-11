# using the skimage package to add noise for 51 subjects
import numpy as np
import os
import skimage
from PIL import Image
import pandas as pd

picpath = 'D:/VGGface2/overlap_vggface1_2'
f = pd.read_csv('D:/cnnface/female_male_test_51_addnoise/51.csv', skiprows=1)
img_path = np.array(f['stimID'])

var = [0.03,0.04]

for img_p in img_path:
    picimg = Image.open(os.path.join(picpath, img_p))
    pic_arr = np.array(picimg)
    subid = img_p.split('/')[0]
    suffix = img_p.split('/')[1]

    for v in var:
        noise_gs_img = skimage.util.random_noise(pic_arr, mode='gaussian', clip=False, var=v)
        noise_gs_img = Image.fromarray((noise_gs_img * 255).astype('uint8'))
        noise_gs_img.save('D:/cnnface/female_male_test_51_addnoise/noise_{}/{}/{}'.format(v, subid, suffix))