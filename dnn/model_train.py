# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:29:57 2020

@author: qyk
"""
import time
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from cnnface.dnn.io import PicDataset
from cnnface.dnn.model_reconstruct import Vgg_identity,Alexnet_gender


def dnn_train_model(dataloaders, model, criterion, optimizer, scheduler, num_epoches=200, train_method='tradition'):
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

    loss_dev = []
    acc_dev = []
    for epoch in range(num_epoches):
        print('Epoch time {}/{}'.format(epoch+1, num_epoches))
        print('-'*10)
        running_loss = 0.0
        running_corrects = 0.0

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
            running_corrects += torch.sum(pred == targets.data)

        #scheduler.step()
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        loss_dev.append(epoch_loss)
        acc_dev.append(epoch_acc)
        print('Loss: {}\n'.format(epoch_loss))
        print('Accuracy: {}\n'.format(epoch_acc))
        time_elapsed = time.time() - time0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - time0
    print('Total training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, loss_dev, acc_dev

# prepare data
images_path = r'D:\cnnface\analysis_for_reply_review\data\train.csv'

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])
dataSet = PicDataset(images_path, transforms)
dataloader = DataLoader(dataSet, batch_size=16, shuffle=True)

# initialize the model,loss function,optimizer
model_train = Vgg_identity()
model_train.load_state_dict(torch.load(r'F:/Code/pretrained_model/vgg_identity_ori.pth'))
optimizer = torch.optim.Adam(model_train.parameters(), lr=0.03)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
loss_func = nn.CrossEntropyLoss()

# train dnn model
# save the parameters and the trainning curve
trained_model,loss_dev,acc_dev = dnn_train_model(dataloader, model_train, loss_func, optimizer, exp_lr_scheduler,
                                                 num_epoches=25)
torch.save(trained_model.state_dict(), r'F:\Code\pretrained_model\review_version\previous_method/vggface_gender.pth')
# np.save(r'D:\cnnface\analysis_for_reply_review\train\vggface\train_dev/loss_dev_nonor_wdecay.npy', loss_dev)
# acc_dev = [acc.cpu().numpy() for acc in acc_dev]
np.save(r'D:\cnnface\analysis_for_reply_review\train&evaluate\train_dev\vggface/acc_dev_previous_method.npy', acc_dev)