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

    # load model
    # vggid = Vgg_identity()
    # vggid.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_gender_CrossEntro.pth'))
    alexnet_gender = Alexnet_gender()
    alexnet_gender.load_state_dict(torch.load('F:/Code/pretrained_model/alexnet_gender_CrossEntro.pth'))


    # load data
    imgcsv_path = r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face\human.csv'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    PicSet = PicDataset(imgcsv_path, transform)
    Picloader = DataLoader(PicSet, batch_size=16, shuffle=False)

    # Get Classification result and activaiton of dnn
    label, label_prob, dnn_act = dnn_ouput(Picloader, alexnet_gender)

    # save Classification result and classification probability
    # np.save(r'D:\cnnface\gender_analysis\supplementray analysis\morph_face_output/morphface_label', label)
    # np.save(r'D:\cnnface\gender_analysis\supplementray analysis\morph_face_output/morphface_prob', label_prob)
    # np.save(r'D:\cnnface\gender_analysis\supplementray analysis\morph_face_output/morphface_act', dnn_act)

    morpface_result = pd.DataFrame({'label':label,
                                    'female_probability':label_prob[:,0],
                                    'male_probability':label_prob[:,1],
                                    'female_activation':dnn_act[:,0],
                                    'male_activation':dnn_act[:,1]})
    morpface_result.to_csv(r'D:\cnnface\gender_analysis\supplementray_analysis\reconstruct_face/vgg_result.csv')