import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset
from dnnbrain.dnn.models import dnn_train_model,Vgg_face

#constrcture vgg_identity model
class Vgg_identity(Vgg_face):
    def __init__(self):
        super().__init__()
        self.fc8 = nn.Linear(4096,2)

#prepare data
images_path = 'D:/cnnface/female_crossEntropLoss_train.csv'
mean = [129.186279296875, 104.76238250732422, 93.59396362304688]
std = [1, 1, 1]
transforms = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                 transforms.Normalize(mean,std)])
dataSet = PicDataset(images_path, transforms, crop=True)

#define the model,loss function,optimizer
vggI = Vgg_identity()
vggI.load_state_dict(torch.load('F:/Code/model/vgg_identity_ori.pth'))

optimizer = torch.optim.Adam(vggI.parameters(),lr = 0.03)
loss_func = nn.CrossEntropyLoss()

#train dnn model
dataloader = DataLoader(dataSet,batch_size=8,shuffle=True)
trained_model = dnn_train_model(dataloader,vggI,loss_func,optimizer,num_epoches=1)
#torch.save(trained_model,'vgg_identity_CrossEntro.pth')