import time
import numpy as np
import torch

def dnn_ouput(dataloaders,model):
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
    label = np.array(label)
    label_prob = np.array(label_prob)
    dnn_act = np.array(dnn_act)
    time_elapsed = time.time() - time0
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return label, label_prob, dnn_act