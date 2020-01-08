import os
import numpy as np
import skimage
import imagehash
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn import metrics as mr
from tqdm import tqdm


def nor(img_arr):
    """ normalize 2d image"""
    return (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())


def img_freq_hist_plot(image_arr, gray=False):
    # plot the frequency histogram of pixel values of image

    if gray:
        image_array = image_arr.reshape(-1)
        image_array = image_array[np.nonzero(image_array)]
        sns.distplot(image_array)
        plt.show()
    if not gray:
        img_r = image_arr[:, :, 0].reshape(-1)
        img_g = image_arr[:, :, 1].reshape(-1)
        img_b = image_arr[:, :, 2].reshape(-1)

        for img, color in zip((img_r, img_g, img_b), ['salmon', 'lawngreen', 'skyblue']):
            sns.distplot(img, color=color)
        plt.legend(['red', 'green', 'blue'])
        plt.show()


def image_power_spectrum(Image_path):
    # plot the power spectrum of image
    pic = Image.open(Image_path)
    data = np.array(pic)
    ps = np.abs(np.fft.fft(data))**2
    time_step = 1 / 50
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx])


def img_add_gaussian_noise(image,std):
    # using the skimage package to add white noise in batch
    # Image[PIL]: Image is PIL image object.
    # var[float]: standard deviation of gaussian noise. range: 0 ~ 1

    image_arr = np.array(image)
    var = std ** 2
    noise_gs_img = skimage.util.random_noise(image_arr,mode='gaussian',clip=True, var=var)
    noise_gs_img = Image.fromarray((noise_gs_img*255).astype('uint8'))
    return noise_gs_img


def img_similarity(img1, img2, method):
    """
    Calculate the similarity of two images

    Parameters
    -----------------------------------------
    :param img1: 2d array of image
    :param img2: 2d array of image
    :param method: [string], the method to calculate the similarity
                    'pearson',pearson correlation coefficient
                    'SSIM', Structural Similarity Index
                    'MI', Mutual Information
                    'cosin', sis
                    'dhash'  Hash value
    Return:
    ----------------------------------------
    similarity[dict]： the dictionary contains many similarity index.

    """
    if img1.shape != img2.shape:
        print('The image size should be the same.')

    if method == 'pearsonr':
        img1 = img1.reshape(-1)
        img2 = img2.reshape(-1)
        similarity = stats.pearsonr(img1, img2)
    elif method == 'spearmanr':
        img1 = img1.reshape(-1)
        img2 = img2.reshape(-1)
        similarity = stats.spearmanr(img1, img2)
    elif method == 'SSIM':
        ssim = skimage.measure.compare_ssim(img1, img2)
        similarity = ssim
    elif method == 'MI':
        img1 = img1.reshape(-1)
        img2 = img2.reshape(-1)
        mi = mr.mutual_info_score(img1, img2)
        similarity = mi
    elif method == 'dhash':
        hash1 = imagehash.dhash(img1)
        hash2 = imagehash.dhash(img2)
        dhash = hash1 - hash2
        similarity = dhash
    else:
        print('The method has not be supported. please input pearson or SSIM or MI or dhash')
    return similarity


def average_imgs(picpathSet):
    images = []
    for path in picpathSet:
        image = np.array(Image.open(path))
        images.append(image)
    imgs = np.array(images)
    aver_img = np.sum(imgs, axis=0) / imgs.shape[0]
    return aver_img


def cal_imgs_ms(images_path):
    """calculate mean and std of the input images
    image_path[str]: images' folder path

    """
    img_filenames = os.listdir(images_path)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(images_path + '/' + img_filename)
        img = img
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)[0][::-1]
    s = s_array.mean(axis=0, keepdims=True)[0][::-1]
    return m, s