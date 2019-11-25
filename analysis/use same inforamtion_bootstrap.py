import os
import numpy as np
from cnnface.stimulus.Image_process import img_similarity,nor
from cnnface.analysis.generate_ci import generateCI, cal_ci
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load ci
ci_human_sum = nor(np.load(r'D:\cnnface\gender_analysis\human_result\CIs\subject/ci_sum.npy'))

# ci_sub_path = 'D:\cnnface\gender_analysis\human_result\CIs\subject\ci'
# ci_sub_list = os.listdir(ci_sub_path)
# ci_human_subjects = []
# for ci in ci_sub_list:
#     ciPath = os.path.join(ci_sub_path,ci)
#     ci_human_subject = nor(np.load(ciPath))
#     ci_human_subjects.append(ci_human_subject)

# generate the 100 CIs from human experiment data
label_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/label_sumxx.npy')
param_exp = np.load(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/param_expxx.npy')
ci_exp = []
for i in range(1000):
    indexList = np.arange(0,9999)
    randomIndex = np.random.choice(indexList, 5000)
    label_rand = label_exp[randomIndex]
    param_rand = param_exp[randomIndex]
    param_ci = cal_ci(param_rand,label_rand)
    ci = nor(generateCI(param_ci))
    ci_exp.append(ci)

# generate the
label_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\label/gender_label_20000.npy')
param_cnn = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
ci_cnn = []
for i in range(1000):
    indexList = np.arange(0,19999)
    randomIndex = np.random.choice(indexList, 5000)
    label_rand = label_cnn[randomIndex]
    param_rand = param_cnn[randomIndex]
    param_ci = cal_ci(param_rand,label_rand)
    ci = nor(generateCI(param_ci))
    ci_cnn.append(ci)

human2template = [img_similarity(ci_human_sum, ci, 'pearson') for ci in ci_exp]
cnn2template = [img_similarity(ci_human_sum, ci, 'pearson') for ci in ci_cnn]

sns.set_style('darkgrid')
sns.distplot(human2template,label='human')
sns.distplot(cnn2template,label='cnn')
plt.show()

s, p = stats.ttest_ind(human2template,cnn2template)

np.save('D:\cnnface\gender_analysis\CI_analysis\same or different/s_p',(s,p))
np.save('D:\cnnface\gender_analysis\CI_analysis\same or different/ci_cnn',ci_cnn)
np.save('D:\cnnface\gender_analysis\CI_analysis\same or different/ci_exp',ci_exp)

ci_cnn_sum = nor(np.load(r'D:\cnnface\gender_analysis\CI_analysis\CIs_img/ci_20000.npy'))

human2template = [img_similarity(ci_cnn_sum, ci, 'pearson') for ci in ci_exp]
cnn2template = [img_similarity(ci_cnn_sum, ci, 'pearson') for ci in ci_cnn]

sns.set_style('darkgrid')
sns.distplot(human2template,label='human')
sns.distplot(cnn2template,label='cnn')
plt.show()

s, p = stats.ttest_ind(human2template,cnn2template)