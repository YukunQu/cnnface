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


if __name__=='__main__':

    from torchvision import transforms
    from dnnbrain.dnn.io import PicDataset, DataLoader
    from cnnface.dnn.model_reconstruct import Vgg_identity

    # load model
    vggid = Vgg_identity()
    vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_identiy2_CrossEntro.pth'))

    # load data
    imgcsv_path = r'D:\cnnface\Emotion_analysis/noiseface_neu.csv'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    PicSet = PicDataset(imgcsv_path, transform)
    Picloader = DataLoader(PicSet, batch_size=16, shuffle=False)

    # Get Classification result and activaiton of dnn
    label, label_prob, dnn_act = dnn_ouput(Picloader, vggid)

    # Get and save Classification result of stimuli
    label_0 = np.argwhere(label == 0).astype('int64')
    label_1 = np.argwhere(label == 1).astype('int64')
    print('Number of label_0:', label_0.shape)
    print('Number of label_1:', label_1.shape)
    np.savetxt('D:\cnnface\Emotion_analysis\CI_analysis/neu_label_happy.txt', label_0)
    np.savetxt('D:\cnnface\Emotion_analysis\CI_analysis/neu_label_sad.txt', label_1)

    # save classification probability of stimuli
    np.save(r'D:\cnnface\female_male_test_51_addnoise\frame054/label_prob', label_prob)