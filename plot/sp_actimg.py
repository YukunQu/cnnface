import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from cnnface.analysis.generate_ci import generateCI,cal_paramci
# Randomly generated ci

param_n = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
label = np.load(r'D:\cnnface\Data_sorted\vggface\raw/gender_label_20000.npy')

trial_num = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,
             3000,4000,5000,6000,8000,10000,12000,14000,16000,18000,20000]

trial_num = [13,17]
saveprepath = r'F:\研究生资料库\项目五：AI\大会报告\actimg/ci_{}.jpg'

for num in trial_num:
    label_random = label[:num]
    param_random = param_n[:num]
    print(num)
    if num == 1:
        ci = generateCI(param_n[0])
    else:
        param_ci = cal_paramci(param_random, label_random)
        ci = generateCI(param_ci)
    plt.imshow(ci,cmap='jet')
    plt.axis('off')
    plt.title(num, y=-0.2,size=28)
    savepath = saveprepath.format(str(num).zfill(5))
    plt.savefig(savepath,bbox_inches='tight', pad_inches=0, dpi=100)

#%%
import os
import imageio


def create_gif(gif_name,image_list,  duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:

        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

prepath = r'F:\研究生资料库\项目五：AI\大会报告\actimg/'
img_name = os.listdir(prepath)
image_list = [prepath+i for i in img_name]
git_save_path = r'F:\研究生资料库\项目五：AI\大会报告\actimg/actimg_ci.gif'
create_gif(git_save_path, image_list, 0.3)
