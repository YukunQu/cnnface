
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from cnnface.dnn.io import PicDataset
from cnnface.dnn.model_reconstruct import Vgg_identity, Alexnet_gender


def train_model(model, dataloaders,dataset_sizes,criterion, optimizer, scheduler, num_epochs=25,):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc_dev = []
    train_loss_dev = []
    val_acc_dev = []
    val_loss_dev = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_dev.append(epoch_loss)
                train_acc_dev.append(epoch_acc.cpu().numpy())
            else:
                val_loss_dev.append(epoch_loss)
                val_acc_dev.append(epoch_acc.cpu().numpy())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # package the dev:
    acc_dev = {'train': train_acc_dev, 'val': val_acc_dev}
    loss_dev = {'train': train_loss_dev, 'val': val_loss_dev}

    return model, acc_dev, loss_dev


####  load data  #####
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
}

train_data_dir = r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\data\train.csv'
val_data_dir = r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\data\val.csv'
image_datasets = {'train': PicDataset(train_data_dir, data_transforms['train']),
                  'val': PicDataset(val_data_dir, data_transforms['val'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_train = Alexnet_gender()
model_train.load_state_dict(torch.load(r'F:/Code/pretrained_model/ori/alexnet_gender_ori.pth'))
# model_train = Alexnet_gender()
# model_train.load_state_dict(torch.load(r'F:/Code/pretrained_model/ori/alexnet_gender_ori.pth'))
model_train = model_train.to(device)
optimizer = torch.optim.Adam(model_train.parameters(), lr=0.03)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_func = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
model_trained, acc_dev, loss_dev = train_model(model_train, dataloaders, dataset_sizes, loss_func, optimizer, exp_lr_scheduler,
                       num_epochs=25)

torch.save(model_trained.state_dict(), r'F:\CAS_PEAL_dataset\model\param/alexnet_gender.pth')
np.save(r'F:\CAS_PEAL_dataset\model\train_dev/alexnet_acc_dev.npy', acc_dev)
np.save(r'F:\CAS_PEAL_dataset\model\train_dev/alexnet_loss_dev.npy', loss_dev)