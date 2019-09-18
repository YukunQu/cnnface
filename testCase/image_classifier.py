import torch
from torchvision import transforms

from dnnbrain.dnn.models import dnn_test_model
from dnnbrain.dnn.io import PicDataset,NetLoader,DataLoader
from cnnface.core.vgg_identity_recons import Vgg_identity

vggid = Vgg_identity()
vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_male_female_CrossEntro.pth'))

imgcsv_path = 'D:\cnnface/female_male_test_51_addnoise/rcicr_gabor_noise.csv'
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
PicSet = PicDataset(imgcsv_path,transform)
Picloader = DataLoader(PicSet,batch_size=32)

output = dnn_test_model(Picloader,vggid)