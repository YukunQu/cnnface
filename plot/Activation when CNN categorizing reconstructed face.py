import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# load the data of activation when cnn categorizing reconstructed face
actHumanFace = np.load(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/human_reconface_act.npy')
actCNNFace = np.load(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/cnn_reconface_act.npy')

# the first figure : bar plot x = scale, y = human_male_act, and cnn_male_act
actHumanMale = actHumanFace[5:, 1]
actCNNMale = actCNNFace[5:, 1]

actHumanFemale = actHumanFace[:5, 0]
actCNNFemale = actCNNFace[:5, 0]

act = np.concatenate((actHumanMale,actCNNMale))
category = ['Human'] * 5 + ['CNN'] *5
Scale = [2,4,8,16,32] *2


sns.set_style('darkgrid')
sns.barplot(x=Scale,y=act,hue=category)
plt.show()