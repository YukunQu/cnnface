import numpy as np
from cnnface.stimuli.image_manipulate import img_similarity


class ClassImage(object):
    """
    A class iamge obejct for ci analysis.

    """

    def __init__(self, data=None):
        """
        Parameter:
            data[2D-array]: The Class image contains the features to classify two types of images.
            parameter[1D-array]: 4092 parameters determined the class image.
        """
        if len(data.shape) == 2:
            self.data = data
        self.parameter = None

    def cal_param(self, paramNtrial, labelNtrial):
        """
        Calculate the ci from noise parameters of n trails and classification label

        Parameter:
        ---------------------------------------------------------------------------
        param_n[array]: 2D array, shape: (trials x 4092). 4092 is parameters for generating the noise
        label[array]: 1D array, shape:(trails,) The label contains classification result of dnn
        Return：
        ---------------------------------------------------------------------------
        param_ci[array]:1D array
        """

        label_0 = np.argwhere(labelNtrial == 0).astype('int32')
        label_1 = np.argwhere(labelNtrial == 1).astype('int32')

        param_0 = paramNtrial[label_0]
        param_1 = paramNtrial[label_1]

        # average the parameters after labeling
        param_0 = np.mean(param_0, axis=0)
        param_1 = np.mean(param_1, axis=0)

        # calculate the CI image from averaging
        self.parameter = np.squeeze(param_0 - param_1)
        return self.parameter

    def generate(self, level='sum', patch='default'):
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
        patchParam = self.parameter[(patchIdx - 1).reshape(-1)].reshape(patchIdx.shape)
        ci_multilevel = patches * patchParam

        if level == 'sum':
            self.data = np.sum(ci_multilevel, axis=2)
        elif isinstance(level, (tuple,list)):
            ci = []
            slice_index = {2: 0, 4: 12, 8: 24, 16: 32, 32: 48}
            for i in level:
                index = slice_index[i]
                ci_level = np.sum(patches[:, :, index:index+12] * patchParam[:, :, index:index+12], axis=2)
                ci.append(ci_level)
            self.data = np.array(ci)
        else:
            print('The level should be the list of space frequency.')
        return self.data


def correlation_ci(ci1,ci2,frequency_scale=False):
    assert ci1.shape == ci2.shape
    if frequency_scale:
        correlation = {}
        scales = [2,4,8,16,32]
        for c1,c2,scale in zip(ci1,ci2,scales):
            correlation[scale] = img_similarity(c1,c2,'pearsonr')
    else:
        correlation = img_similarity(ci1,ci2,'pearsonr')
    return correlation
