# Get label of Noise images using classification of dnn
import numpy as np
import torch
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset, DataLoader
from cnnface.dnn.vgg_identity_recons import Vgg_identity
from dnnbrain.dnn.models import dnn_test_model

# load noise image
imgcsv_path =  'D:\cnnface\Identity_analysis/bf_noise5000.csv'
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
PicSet = PicDataset(imgcsv_path, transform)
Picloader = DataLoader(PicSet, batch_size=32,shuffle=False)
# load model
vggid = Vgg_identity()
vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_identity2_CrossEntro.pth'))

# Get Classification result of vgg
label, _, _ = dnn_test_model(Picloader, vggid)
label_0 = np.argwhere(label == 0).astype('int64')
label_1 = np.argwhere(label == 1).astype('int64')
print('Number of label_0:', label_0.shape)
print('Number of label_1:', label_1.shape)

# Get and save Classification result
np.savetxt('D:\cnnface\Identity_analysis/label_ac.txt', label_0)
np.savetxt('D:\cnnface\Identity_analysis/label_nc.txt', label_1)
