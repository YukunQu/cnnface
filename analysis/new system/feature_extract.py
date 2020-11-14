import numpy as np
from PIL import Image
from cnnface.analysis.generate_ci import generateCI
from cnnface.stimuli.image_manipulate import nor


def ci2weight(ci_path):
    ci = np.load(ci_path)
    img = Image.fromarray(ci)
    ci_resize = np.array(img)
    #  ci_scaled= (ci_resize*100)**5
    weight = ci_resize
    print(weight)
    print(weight.shape)
    return weight


def load_image_arr(img_path):
    return np.array(Image.open(img_path).convert('L'))


def feature_extract(img_arr, weight, scale=None):
    return img_arr * weight


def feature_save(feature, savepath):

    img = Image.fromarray(feature).convert('RGB')
    img.save(savepath)


def extract_ci_fromimg(img_path, savepath):
    ci_path = r'D:\cnnface\gender_analysis\CI_analysis/ci_cnn.npy'
    weight = ci2weight(ci_path)
    img_arr = load_image_arr(img_path)
    feature = feature_extract(img_arr, weight)
    return feature
    #feature_save(feature,savepath)


def generate_core_feature(param_n, label):
    """
    Calculate the ci from noise parameters of n trails and classification label

    Parameter:
    ---------------------------------------------------------------------------
    param_n[array]: 2D array, trials x 4092. 4092 is parameters for generating the noise
    label[array]: 1D array, shape:(trails,) The label contains classification result of dnn
    subjectid[list], The parameter works with subjtrials. The list contains subject id to calculate the paramci.
    Return：
    ---------------------------------------------------------------------------
    param_ci[array]:1D array
    """
    label_0 = np.argwhere(label == 0).astype('int32')
    label_1 = np.argwhere(label == 1).astype('int32')

    param_0 = param_n[label_0]
    param_1 = param_n[label_1]

    # average the parameters after labeling
    param_0 = np.squeeze(np.mean(param_0, axis=0))
    param_1 = np.squeeze(np.mean(param_1, axis=0))

    female_feature = generateCI(param_0)
    male_feature = generateCI(param_1)

    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\core_feature/female_feature', female_feature)
    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\core_feature/male_feature', male_feature)

    return female_feature, male_feature


if __name__=="__main__":
    img_path = r'D:\cnnface\analysis_for_reply_review\data\registrated\train\male/1.png'
    savepath = r'D:\cnnface\analysis_for_reply_review\data\feature_extract/male_1.png'
    feature = extract_ci_fromimg(img_path, None)
    #extract_ci_fromimg(img_path, savepath)

