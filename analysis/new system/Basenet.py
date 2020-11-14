import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from cnnface.dnn.io import PicDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from cnnface.dnn.model_train import dnn_train_model
from cnnface.dnn.model_test import dnn_test_model


class Basenet(nn.Module):
    def __init__(self):
        super(Basenet, self).__init__()
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y


def basenet_train(stim_path, param_save_path, acc_save_path):
    # prepare data
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    dataSet = PicDataset(stim_path, transform, img_mode='L')
    dataloader = DataLoader(dataSet, batch_size=16, shuffle=True)

    # load model
    model_train = Basenet()
    # set train super parameters
    optimizer = torch.optim.Adam(model_train.parameters(), lr=0.03)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()

    # train dnn model
    trained_model, loss_dev, acc_dev = dnn_train_model(dataloader, model_train, loss_func, optimizer, exp_lr_scheduler,
                                                       num_epoches=20)
    torch.save(trained_model.state_dict(), param_save_path)
    np.savetxt(acc_save_path, acc_dev)


def basenet_test(stim_path, param_path, result_dir):
    # prepare data
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    dataSet = PicDataset(stim_path, transform, img_mode='L')
    dataloader = DataLoader(dataSet, batch_size=16, shuffle=False)

    # load model
    model_test = Basenet()
    model_test.load_state_dict(torch.load(param_path))

    # model test
    # save the parameters and the trainning curve
    test_result = dict()
    test_result['label'], test_result['expect_label'], test_result['accuracy'] = dnn_test_model(dataloader, model_test)
    print("Test accuracy:", test_result['accuracy'])
    np.save(result_dir + r'/test_result.npy', test_result)


if __name__ == "__main__":

    # stim_path = r'D:\cnnface\analysis_for_reply_review\data\registrated/train.csv'
    # param_path = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet/param/cinet_baseline.pth'
    # result_dir = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet\train&test\baseline\train/acc_dev.txt'
    # basenet_train(stim_path, param_path, result_dir)

    stim_path = r'D:\cnnface\analysis_for_reply_review\data\registrated/val.csv'
    param_path = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet/param/cinet_baseline.pth'
    result_dir = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet\train&test\baseline/val'
    basenet_test(stim_path, param_path, result_dir)