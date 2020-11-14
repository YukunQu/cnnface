import time
import numpy as np
import torch


def dnn_ouput(dataloaders, model):
    """
    Get model output and classification probability.

    Parameters:
    -----------
    dataloaders[dataloader]: dataloader generated from dataloader(PicDataset)
    model[class/nn.Module]: DNN model with pretrained parameters

    Returns:
    --------
    label[array]: classification label of model
    label_prob[array]: classification probability of label
    dnn_act[array]: activation of dnn
    """
    time0 = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    label = []
    label_prob = []
    dnn_act = []
    with torch.no_grad():
        for i, (picbatch, condition) in enumerate(dataloaders):
            print('Now loading batch {}'.format(i + 1))
            picbatch = picbatch.to(device)
            dnn_act_part = model(picbatch)
            _, label_part = torch.max(dnn_act_part, 1)
            label_prob_part = torch.softmax(dnn_act_part, 1)
            dnn_act.extend(dnn_act_part.cpu().numpy())
            label.extend(label_part.cpu().numpy())
            label_prob.extend(label_prob_part.cpu().numpy())
    label = np.squeeze(np.array(label))
    label_prob = np.squeeze(np.array(label_prob))
    dnn_act = np.squeeze(np.array(dnn_act))
    time_elapsed = time.time() - time0
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return label, label_prob, dnn_act


if __name__=='__main__':
    import pandas as pd
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from cnnface.dnn.io import PicDataset
    from cnnface.dnn.model_reconstruct import Vgg_identity, Alexnet_gender
    from torch import nn
    from torchvision.models import vgg16

    #load model
    # model_gender = Vgg_identity()
    # model_gender.load_state_dict(torch.load(r'F:\Code\pretrained_model\review_version\2.0/vggface_gender.pth'))

    model_gender = Vgg_identity()
    model_gender.load_state_dict(torch.load(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\model\parameter/vggface_gender.pth'))

    # load data
    imgcsv_path = r'D:\cnnface\gender_analysis\noise_stimulus/baseface_20000.csv'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    PicSet = PicDataset(imgcsv_path, transform)
    Picloader = DataLoader(PicSet, batch_size=16, shuffle=False)

    # Get Classification result and activaiton of dnn
    label, label_prob, dnn_act = dnn_ouput(Picloader, model_gender)

    # # save Classification result and classification probability
    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\alexnet/label.npy', label)
    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\alexnet/prob.npy', label_prob)
    np.save(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\alexnet/act.npy', dnn_act)

    # #%%
    # male_prob = label_prob[:, 1]
    # distance = np.abs(male_prob - 0.5)
    # minIndex = np.argwhere(distance == distance.min())
    # np.save(r'D:\AI_generate_faces\morphface_act/minIndex.npy', minIndex)