import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from cnnface.dnn.io import PicDataset
from cnnface.dnn.model_reconstruct import Vgg_identity
from cnnface.dnn.model_reconstruct import Alexnet_gender
from torchvision.models import vgg16

#%%
def dnn_train_model(dataloaders, model, criterion, optimizer, num_epoches=200, train_method='tradition'):
    """
    Function to train a DNN model

    Parameters:
    ------------
    dataloaders[dataloader]: dataloader generated from dataloader(PicDataset)
    model[class/nn.Module]: DNN model without pretrained parameters
    criterion[class]: criterion function
    optimizer[class]: optimizer function
    num_epoches[int]: epoch times, by default is 200.
    train_method[str]: training method, by default is 'tradition'.
                       For some specific models (e.g. inception), loss needs to be calculated in another way.

    Returns:
    --------
    model[class/nn.Module]: model with trained parameters.
    """
    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    for epoch in range(num_epoches):
        print('Epoch time {}/{}'.format(epoch+1, num_epoches))
        print('-'*10)
        running_loss = 0.0
        loss_change_curve = []
        for inputs, targets in dataloaders:
            inputs.requires_grad_(True)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if train_method == 'tradition':
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                elif train_method == 'inception':
                    # Google inception model
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, targets)
                    loss2 = criterion(aux_outputs, targets)
                    loss = loss1 + 0.4*loss2
                else:
                    raise Exception('Not Support this method yet, please contact authors for implementation.')

                _, pred = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            # Statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloaders.dataset)
        loss_change_curve.append(epoch_loss)
        print('Loss: {}\n'.format(epoch_loss))
    time_elapsed = time.time() - time0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, loss_change_curve


#prepare data
images_path = r'D:\cnnface\gender_analysis\train_stimulus\train.csv'
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
dataSet = PicDataset(images_path, transforms)

# initialize the vggface model,loss function,optimizer
# vggI = Vgg_identity()
# vggI.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_identity_ori.pth'))
#
# optimizer = torch.optim.Adam(vggI.parameters(), lr=0.03)

# initialize the vgg16 model,loss function,optimizer
vgg16_gender = vgg16()
for param in vgg16_gender.parameters():
    param.requires_grad = False
vgg16_gender.classifier[-1] = nn.Linear(4096, 2, bias=True)
vgg16_gender.load_state_dict(torch.load(r'F:/Code/pretrained_model/vgg16_gender_ori.pth'))
#optimizer = torch.optim.Adam(vgg16_gender.parameters(), lr=0.01)
optimizer = torch.optim.SGD(vgg16_gender.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()
for name, param in vgg16_gender.named_parameters():
    print("{} {}".format(name, param.requires_grad))

# train alexnet_gender network
# alexnet_gender = Alexnet_gender()
# alexnet_gender.load_state_dict(torch.load('F:/Code/pretrained_model/alexnet_gender_ori.pth'))
#
# optimizer = torch.optim.Adam(alexnet_gender.parameters(), lr=0.03)
# loss_func = nn.CrossEntropyLoss()


# train dnn model
dataloader = DataLoader(dataSet, batch_size=16, shuffle=True)
trained_model, loss_curve = dnn_train_model(dataloader, vgg16_gender, loss_func, optimizer, num_epoches=25)
torch.save(trained_model.state_dict(), 'F:/Code/pretrained_model/vgg16_gender_CrossEntro_sgd_lr0.01.pth')
np.save(r'D:\cnnface\Data_sorted\vgg16\train/loss_change_curve_sgd_lr0.01.npy', loss_curve)