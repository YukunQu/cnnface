import numpy as np


def cal_ci(param_n, label):
    """
    Calculate the ci from noise parameters of n trails and classification label

    Parameter:
    ---------------------------------------------------------------------------
    param_n[array]: 2D array, trials x 4092. 4092 is parameters for generating the noise
    label[array]: 1D array, shape:(trails,) The label contains classification result of dnn

    Return：
    ---------------------------------------------------------------------------
    param_ci[array]:1D array
    """

    label_0 = np.argwhere(label == 0).astype('int32')
    label_1 = np.argwhere(label == 1).astype('int32')

    param_0 = param_n[label_0]
    param_1 = param_n[label_1]

    # average the parameters after labeling
    param_0 = np.mean(param_0, axis=0)
    param_1 = np.mean(param_1, axis=0)

    # calculate the CI image from averaging
    param_ci = param_0 - param_1
    return param_ci


def generateCI(param, patch='default', level='all', scale=None):
    """
    Generate the classification image from parameters of ci.

    Parameter:
    --------------------------------------------------------
    param[array]: len:4092. parameters of ci
    patch[dict]: patch pattern for generating the ci from generate_noise_img
                patches: 3D array width x high x 60
                patchIdx 3D array width x high x 60
    level[tuple/list]: The list of space frequency for getting the ci of specific space frequency

    Return:
    --------------------------------------------------------
    ci[array] 2D array or the list of 2D array contains different levels of ci
    """
    # load patches and patchidx
    if patch == 'default':
        patches = np.load(r'D:\cnnface/patches.npy')
        patchIdx = np.load(r'D:\cnnface/patchidx.npy').astype('int32')
    else:
        patches = patch['patches']
        patchIdx = patch['patchIdx']
    # generate ci image
    patchParam = param[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)
    ci_multilevel = patches * patchParam

    if level == 'all':
        ci = np.sum(ci_multilevel, axis=2)
    elif isinstance(level, tuple) or isinstance(level, list):
        ci = []
        slice_index = {2: 0, 4: 12, 8: 24, 16: 32, 32: 48}
        for i in level:
            index = slice_index[i]
            ci_level = np.sum(patches[:, :, index:index+12] * patchIdx[:, :, index:index+12], axis=2)
            ci.append(ci_level)
    else:
        print('The level should be the list of space frequency.')
    return ci
