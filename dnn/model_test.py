# 对训练的网络进行测试
import numpy as np
import torch
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset, DataLoader
from cnnface.dnn.model_reconstruct import Vgg_identity
from dnnbrain.dnn.models import dnn_test_model

# load model
vggid = Vgg_identity()
vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_gender_CrossEntro.pth'))

# load noise image
imgcsv_path =  r'D:\cnnface\gender_analysis\train_stimulus\test.csv'
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
PicSet = PicDataset(imgcsv_path, transform)
Picloader = DataLoader(PicSet, batch_size=16, shuffle=False)

# Get Classification result of vgg
label, expect_label, accuracy = dnn_test_model(Picloader, vggid)
print(accuracy)