import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#
# # load the data of activation when cnn categorizing reconstructed face
# actHumanFace = pd.read_csv(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/human_reconface_act.npy')
# actCNNFace = pd.read_csv(r'D:\cnnface\Data_sorted\vggface\prototype_face\differentScale_act/vgg_reconface_result.csv')
# # the first figure : bar plot x = scale, y = human_male_act, and cnn_male_act
# actHumanMale = actHumanFace['male_activation'][5:]
# actCNNMale = actCNNFace['male_activation'][5:]
#
# actHumanFemale = actHumanFace['female_activation'][:5]
# actCNNFemale = actCNNFace['female_activation'][:5]
#
# #act = np.concatenate((actHumanFemale,actCNNFemale,actAlexFemale))
# act = np.concatenate((actHumanFemale,actCNNFemale))
# category = ['Human'] * 5 + ['Vgg'] * 5
# Scale = [2,4,8,16,32] * 2
#
#
# sns.set_style('darkgrid')
# sns.barplot(x=Scale, y=act, hue=category)
# plt.title('female')
# plt.savefig(r'D:\cnnface\Data_sorted\compareResult\prototypeface/female_act.jpg',dpi=300)
# plt.show()

# load the data of activation when cnn categorizing reconstructed face
actHumanFace = np.load(r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face/human_reconface_act.npy')
actCNNFace = np.load(r'D:\cnnface\Data_sorted\vggface\prototype_face\differentScale_act/act.npy')

# the first figure : bar plot x = scale, y = human_male_act, and cnn_male_act
actHumanMale = actHumanFace[5:, 1]
actCNNMale = actCNNFace[5:, 1]

actHumanFemale = actHumanFace[:5, 0]
actCNNFemale = actCNNFace[:5, 0]

category = ['Human'] * 5 + ['VGG-Face'] *5
Scale = [2,4,8,16,32] *2

act = np.concatenate((actHumanFemale,actCNNFemale))
sns.barplot(x=Scale,y=act,hue=category)
plt.legend(frameon=False)
plt.title('Unit: female',size=20)
#plt.ylabel('Activation',size=)
plt.yticks([5,10,15,20],['5','10','15','20'],size=16)
plt.xticks(size=16)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\prototypeface/female_act.jpg',dpi=600)
plt.show()

act = np.concatenate((actHumanMale,actCNNMale))
sns.barplot(x=Scale,y=act,hue=category)
plt.legend(frameon=False)
plt.title('Unit: male',size=20)
#plt.ylabel('Activation')
plt.yticks([5,10,15,20],['5','10','15','20'],size=16)
plt.xticks(size=16)
plt.savefig(r'D:\cnnface\Data_sorted\compareResult\prototypeface/male_act.jpg',dpi=600)
plt.show()