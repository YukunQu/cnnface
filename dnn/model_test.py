# 对训练的网络进行测试
import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16
from cnnface.dnn.io import PicDataset
from torch.utils.data import DataLoader
from cnnface.dnn.model_reconstruct import Vgg_identity, Alexnet_gender


def dnn_test_model(dataloaders, model):
    """
    Test model accuracy.

    Parameters:
    -----------
    dataloaders[dataloader]: dataloader generated from dataloader(PicDataset)
    model[class/nn.Module]: DNN model with pretrained parameters

    Returns:
    --------
    model_target[array]: model output
    actual_target [array]: actual target
    test_acc[float]: prediction accuracy
    """
    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    model_target = []
    actual_target = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloaders):
            print('Now loading batch {}'.format(i+1))
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, outputs_label = torch.max(outputs, 1)
            model_target.extend(outputs_label.cpu().numpy())
            actual_target.extend(targets.numpy())
    model_target = np.array(model_target)
    actual_target = np.array(actual_target)
    test_acc = 1.0*np.sum(model_target == actual_target)/len(actual_target)
    time_elapsed =  time.time() - time0
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_target, actual_target, test_acc

# load model

model_gender = Vgg_identity()
model_gender.load_state_dict(torch.load(r'F:\Code\pretrained_model\review_version\previous_stim/vggface_gender.pth'))


# load noise image
imgcsv_path = r'D:\cnnface\analysis_for_reply_review\data/val.csv'
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
PicSet = PicDataset(imgcsv_path, transform)
Picloader = DataLoader(PicSet, batch_size=16, shuffle=False)

# Get Classification result of vgg
label, expect_label, accuracy = dnn_test_model(Picloader, model_gender)
print(accuracy)


