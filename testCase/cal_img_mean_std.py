import cv2, os
import numpy as np
from tqdm import tqdm


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
    return m,s


img_path1 = 'D:/cnnface/femaletrain/n00000068'
img_path2 = 'D:/cnnface/femaletrain/n00001586'

sub1_mean,sub1_std = cal_img_ms(img_path1)
sub2_mean,sub2_std = cal_img_ms(img_path2)

sub_mean = (sub1_mean + sub2_mean)/2
sub_std = (sub1_std + sub2_std)/2

