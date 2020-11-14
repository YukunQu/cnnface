import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from cnnface.dnn.io import PicDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from cnnface.dnn.model_train import dnn_train_model


class Simple_classifier(nn.Module):
    # It is a simple face classifer using Classification image as the weight.
    def __init__(self):
        super(Simple_classifier, self).__init__()
        self.classifier = nn.Linear(50176, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = nn.sigmoid(x)
        return x


def simple_classifier(load_ci):
    model = Simple_classifier()
    if load_ci:
        ci_weight = ci2weight(load_ci)
        model.classifier.weigh = ci_weight
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    return simple_classifier()


def ci2weight(ci_path):
    ci = np.load(ci_path)
    ci_resize = ci.resize((224, 224))
    ci = torch.tensor(ci_resize)
    weight = torch.flatten(ci).unsqueeze(0)
    return weight


if __name__ == "__main__":
    # prepare data
    images_path = r'D:\cnnface\analysis_for_reply_review\data\registrated/train.csv'

    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])
    dataSet = PicDataset(images_path, transforms)
    dataloader = DataLoader(dataSet, batch_size=16, shuffle=True)

    # load model
    ci_path = r'D:\cnnface\gender_analysis\CI_analysis/ci_cnn.npy'
    model_train = simple_classifier(ci_path)

    # set train super parameters
    optimizer = torch.optim.Adam(model_train.parameters(), lr=0.03)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()

    # train dnn model
    # save the parameters and the trainning curve
    trained_model,loss_dev,acc_dev = dnn_train_model(dataloader, model_train, loss_func, optimizer, exp_lr_scheduler,
                                                     num_epoches=20)
    torch.save(trained_model.state_dict(), r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier/'
                                           r'simple_classifier.pth')
    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\simple_classifier/acc_simple_classifier.npy', acc_dev)