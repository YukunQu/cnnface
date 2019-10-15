import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset
from dnnbrain.dnn.models import dnn_train_model
from cnnface.dnn.vgg_identity_recons import Vgg_identity


#prepare data
images_path = 'D:\cnnface\Identity_analysis/identity2_train.csv'
transforms = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
dataSet = PicDataset(images_path, transforms)

#define the model,loss function,optimizer
vggI = Vgg_identity()
vggI.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_identity_ori.pth'))

optimizer = torch.optim.Adam(vggI.parameters(),lr = 0.03)
loss_func = nn.CrossEntropyLoss()

#train dnn model
dataloader = DataLoader(dataSet,batch_size=8,shuffle=True)
trained_model = dnn_train_model(dataloader,vggI,loss_func,optimizer,num_epoches=4)
torch.save(trained_model.state_dict(),'F:/Code/pretrained_model/vgg_identity2_CrossEntro.pth')