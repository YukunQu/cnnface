import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


act = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier/act_baseline.npy')

female_act = act[:13830]
male_act = act[13830:]

label = ['Female'] * 13831 + ["Male"] * 13286
data = pd.DataFrame({'Activation': act, "Label": label})

sns.displot(data, x='Activation', hue='Label', kind="kde", fill=True)
plt.show()
