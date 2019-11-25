import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d_human = np.load(r'D:\cnnface\gender_analysis\human_result\para_significant/cohensd_sum.npy')
d_cnn = np.load(r'D:\cnnface\gender_analysis\CI_analysis\para_significant/cohensd_sum.npy')