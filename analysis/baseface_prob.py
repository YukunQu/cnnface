# ------------------------------------------------------------------------------ #
# generate the classification probability of baseface    #
# ------------------------------------------------------------------------------ #

import numpy as np
import seaborn as sns

import torch
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset, DataLoader
from cnnface.dnn.vgg_identity_recons import Vgg_identity
from cnnface.dnn.dnn_output import dnn_ouput

#load model and data
vggid = Vgg_identity()
vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_male_female_CrossEntro.pth'))  # load model

imgcsv_path =  r'D:\cnnface\female_male_test_51_addnoise/frame054.csv'   # path of base face
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
PicSet = PicDataset(imgcsv_path, transform)
Picloader = DataLoader(PicSet, batch_size=10,shuffle=False)

# get activation
label, label_prob, dnn_act = dnn_ouput(Picloader,vggid)

sns.set_style('darkgrid')
x = np.arange(0,101,1)
y1 = label_prob[:,0]
y2= label_prob[:,1]

np.save(r'D:\cnnface\female_male_test_51_addnoise\frame054/label_prob',label_prob)