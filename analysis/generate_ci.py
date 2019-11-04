import numpy as np
from PIL import Image


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


def generateCI(param,  level='sum', patch='default'):
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

    if level == 'sum':
        ci = np.sum(ci_multilevel, axis=2)
    elif isinstance(level, tuple) or isinstance(level, list):
        ci = []
        slice_index = {2: 0, 4: 12, 8: 24, 16: 32, 32: 48}
        for i in level:
            index = slice_index[i]
            ci_level = np.sum(patches[:, :, index:index+12] * patchParam[:, :, index:index+12], axis=2)
            ci.append(ci_level)
        ci = np.array(ci)
    else:
        print('The level should be the list of space frequency.')
    return ci


def recon_face(baseface, ci, scale=1.0):
    """
    Reconstruct face from baseface and ci

    Parameter:
    ---------------------------------------------------------------------------------------------
    baseface[PIL]: PIL file contains the baseface
    ci[array]: 2D array, contains the classification image which can be generated from generateCI

    Return:
    ---------------------------------------------------------------------------------------------
    img_add,img_sub[PIL]: reconstruct face
    """
    baseface = np.array(baseface).astype('float64')

    ci_scale = ci * scale
    bf_add = baseface + ci_scale
    bf_sub = baseface - ci_scale

    bf_add[bf_add > 255] = 255
    bf_add[bf_add < 0] = 0
    bf_sub[bf_sub > 255] = 255
    bf_sub[bf_sub < 0] = 0

    img_add = Image.fromarray(bf_add.astype('int8')).convert('L')
    img_sub = Image.fromarray(bf_sub.astype('int8')).convert('L')
    return img_add, img_sub


#%%
import numpy as np
from PIL import Image
from cnnface.analysis.generate_ci import generateCI,recon_face

param_ci = np.load(r'D:\cnnface\female_male_test_51_addnoise\Face_template\meta_data/paras_ci.npy')
cis = generateCI(param_ci,level=(2,4,8,16,32))
cis_34 = cis * 34
cis_68 = cis * 68

baseface = Image.open(r'D:\cnnface\female_male_test_51_addnoise\frame054/frame054_gray_512.jpg')
level=(2,4,8,16,32)

for i, l in enumerate(level):
    print(cis_34[i,:,:].shape)
    img_add, img_sub = recon_face(baseface, cis_34[i, :, :])
    img_add.save(r'D:\cnnface\female_male_test_51_addnoise\Face_template\classification_noise\different_level_CI//34/bf_add_%04d.jpg' % l)
    img_sub.save(r'D:\cnnface\female_male_test_51_addnoise\Face_template\classification_noise\different_level_CI//34/bf_sub_%04d.jpg' % l)

for i, l in enumerate(level):
    print(cis_68[i,:,:].max())
    print(cis_68[i, :, :].min())
    img_add, img_sub = recon_face(baseface, cis_68[i, :, :])
    img_add.save(r'D:\cnnface\female_male_test_51_addnoise\Face_template\classification_noise\different_level_CI//68/bf_add_%04d.jpg' % l)
    img_sub.save(r'D:\cnnface\female_male_test_51_addnoise\Face_template\classification_noise\different_level_CI//68/bf_sub_%04d.jpg' % l)


