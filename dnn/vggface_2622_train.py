import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from cnnface.dnn.io import PicDataset
from cnnface.dnn.model_reconstruct import Vgg_face
from twoStream.model.train2 import dnn_train_model


# prepare data
images_path = r'F:\vggface2/train.csv'
transforms = transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.RandomCrop((224, 224)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor()])
dataSet = PicDataset(images_path, transforms)
dataloader = DataLoader(dataSet, batch_size=8, shuffle=True)

# initialize model and set hyper-parameter
vggface = Vgg_face()
optimizer = torch.optim.SGD(vggface.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_func = nn.CrossEntropyLoss()

# train dnn model
trained_model,loss_dev,acc_dev = dnn_train_model(dataloader, vggface, loss_func, optimizer, exp_lr_scheduler, num_epoches=5)
torch.save(trained_model.state_dict(), r'F:\vggface2/model/vggface_crossEntro_epoch5.pth')
np.save(r'F:\vggface2/train_dev/train_loss_dev_crossEntro.npy',loss_dev)
acc_dev = [acc.cpu().numpy() for acc in acc_dev]
np.save(r'F:\vggface2/train_dev/acc_loss_dev_crossEntro.npy',acc_dev)