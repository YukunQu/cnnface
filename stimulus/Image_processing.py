import numpy as np
import skimage
from PIL import Image
import matplotlib.pyplot as plt

import cv2, os
from tqdm import tqdm


def img_r2(img1, img2):
    pass


def image_freq_hist_plot(image_path):
    # plot the frequency histogram of pixel values of image

    image = Image.open(image_path)
    image_array = np.array(image)
    image_array = image_array.reshape(-1)
    plt.hist(image_array, 255)


def image_power_spectrum(Image_path):
    # plot the power spectrum of image
    pic = Image.open(Image_path)
    data = np.array(pic)
    ps = np.abs(np.fft.fft(data))**2
    time_step = 1 / 50
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx])


def image_add_gaussian_noise(Image,std):
    # using the skimage package to add white noise in batch
    # Image[PIL]: Image is PIL image object.
    # var[float]: standard deviation of gaussian noise. range: 0 ~ 1

    image_arr = np.array(Image)
    var = std ** 2
    noise_gs_img = skimage.util.random_noise(image_arr,mode='gaussian',clip=True,var=var)
    noise_gs_img = Image.fromarray((noise_gs_img*255).astype('uint8'))
    return noise_gs_img


def cal_img_ms(image_path):
    """calculate mean and std of the input images
    image_path[str]: images' folder path

    """
    img_filenames = os.listdir(image_path)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(image_path + '/' + img_filename)
        img = img
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)[0][::-1]
    s = s_array.mean(axis=0, keepdims=True)[0][::-1]
    return m, s


def average_img(picpathSet):
    images = []
    for path in picpathSet:
        image = np.array(Image.open(path))
        images.append(image)
    imgs = np.array(images)
    aver_img = np.sum(imgs, axis=0) / imgs.shape[0]
    return aver_img


def img2gray(r_path,s_path):
    img = Image.open(r_path)
    img = img.convert('RGB')
    img.save(s_path,quality=95)

