import numpy as np

past_p = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/p_sum.npy')

p_signIndex = np.argwhere(past_p < (0.05/4092))

p_sum = np.load(r'D:\cnnface\Data_sorted\vggface\param_analysis\new\data/d_sum.npy')
