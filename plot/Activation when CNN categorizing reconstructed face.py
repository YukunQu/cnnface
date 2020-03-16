import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# load the data of activation when cnn categorizing reconstructed face
actHumanFace = pd.read_csv(r'D:\cnnface\gender_analysis\supplementray_analysis\reconstruct_face/human.csv')
actCNNFace = pd.read_csv(r'D:\cnnface\gender_analysis\supplementray_analysis\reconstruct_face/vgg_result.csv')
resultAlexFace = pd.read_csv(r'D:\cnnface\gender_analysis\supplementray_analysis\reconstruct_face/alex_gender_result.csv')
# the first figure : bar plot x = scale, y = human_male_act, and cnn_male_act
actHumanMale = actHumanFace['male_activation'][5:]
actCNNMale = actCNNFace['male_activation'][5:]
actAlexMale = resultAlexFace['male_activation'][5:]

actHumanFemale = actHumanFace['female_activation'][:5]
actCNNFemale = actCNNFace['female_activation'][:5]
actAlexFemale = resultAlexFace['female_activation'][:5]

#act = np.concatenate((actHumanFemale,actCNNFemale,actAlexFemale))
act = np.concatenate((actHumanFemale,actCNNFemale,actAlexFemale))
category = ['Human'] * 5 + ['Vgg'] * 5 + ['Alex'] * 5
Scale = [2,4,8,16,32] * 3


sns.set_style('darkgrid')
sns.barplot(x=Scale, y=act, hue=category)
plt.title('female')
plt.savefig(r'D:\cnnface\gender_analysis\supplementray_analysis\reconstruct_face\result/female_act.jpg')
plt.show()

