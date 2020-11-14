import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from cnnface.dnn.io import PicDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from cnnface.dnn.model_train import dnn_train_model
from cnnface.dnn.model_test import dnn_test_model


class CInet(nn.Module):
    def __init__(self):
        super(CInet, self).__init__()
        self.classifier = nn.Linear(64, 2)

    def feature_extract(self, x):
        masks = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask\alexnet-mask-64/'
                        r'masks.npy')
        masks = torch.tensor(masks)
        masks = torch.flatten(masks, 1).cuda()
        x = torch.flatten(x, 1)
        # print("Input:", x.shape)
        # print("Mask:", masks.shape)
        feature_vector = torch.matmul(x, masks.t())
        #print("Feature vector:",feature_vector.size())
        return feature_vector

    def forward(self, x):
        x = self.feature_extract(x)
        y = self.classifier(x)
        return y


def cinet_test(stim_path, param_path, result_dir):
    # prepare data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataSet = PicDataset(stim_path, transform, img_mode='L')
    dataloader = DataLoader(dataSet, batch_size=16, shuffle=False)

    # load model
    model_test = CInet()
    model_test.load_state_dict(torch.load(param_path))

    # model test
    # save the parameters and the trainning curve
    test_result = dict()
    test_result['label'], test_result['expect_label'], test_result['accuracy'] = dnn_test_model(dataloader, model_test)
    print("Test accuracy:", test_result['accuracy'])
    #np.save(result_dir + r'/test_result_2.0.npy', test_result)



class CInetv3(nn.Module):
    def __init__(self):
        super(CInetv3, self).__init__()
        self.classifier = nn.Linear(2, 2)

    def feature_extract(self, x):
        masks = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask\mask-2/'
                        r'masks.npy')
        masks = masks * 10
        # masks = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\feature_extract_mask\alexnet-mask-64/'
        #                 r'masks.npy')
        masks = torch.tensor(masks)
        masks = torch.flatten(masks, 1).cuda()
        x = torch.flatten(x, 1)
        # print("Input:", x.shape)
        # print("Mask:", masks.shape)
        feature_vector = torch.matmul(x, masks.t())
        #print("Feature vector:",feature_vector.size())
        return feature_vector

    def forward(self, x):
        x = self.feature_extract(x)
        y = self.classifier(x)
        return y


def cinet_train(stim_path, param_save_path, acc_save_path):
    # prepare data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataSet = PicDataset(stim_path, transform, img_mode='L')
    dataloader = DataLoader(dataSet, batch_size=16, shuffle=True)

    # load model
    model_train = CInet()

    # set train super parameters
    optimizer = torch.optim.Adam(model_train.parameters(), lr=0.03)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()

    # train dnn model
    # save the parameters and the trainning curve
    trained_model, loss_dev, acc_dev = dnn_train_model(dataloader, model_train, loss_func, optimizer, exp_lr_scheduler,
                                                       num_epoches=25)
    torch.save(trained_model.state_dict(), param_save_path)
    np.savetxt(acc_save_path, acc_dev)


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


if __name__ == "__main__":
    stim_path = r'D:\cnnface\analysis_for_reply_review\data\registrated/test.csv'
    param_path = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet/param/cinet_alexnet.pth'
    result_dir = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet\train&test\alexnet\test'
    cinet_test(stim_path, param_path, result_dir)

    # stim_path = r'D:\cnnface\analysis_for_reply_review\data\registrated/train.csv'
    # param_path = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet/param/cinet_v4.0.pth'
    # result_dir = r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier\CInet\train&test\vggface\train/' \
    #              r'acc_dev_v4.0.txt'
    # cinet_train(stim_path, param_path, result_dir)