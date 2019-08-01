import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset,DataLoader
from dnnbrain.dnn.models import Vgg_face,dnn_test_model
from cnnface.core.vgg_identity_recons import Vgg_identity
import pandas as pd
import os

def test_dataSet(model, images_path):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    dataSet = PicDataset(images_path, transform, crop=True)
    dataloader = DataLoader(dataSet, batch_size=8, shuffle=False)
    model_target, actual_target, _ = dnn_test_model(dataloader, model)

    return model_target,actual_target


def test_onepicture(model,image_path,crop_coord):
    image = Image.open(image_path)
    image = image.crop(crop_coord)
    image.save('D:/cnnface/femaletrain/crop.jpg')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)

    for i in range(10):
        classifier_act = model(image)
        classifier_act = classifier_act.squeeze(0)
        classifier_num = classifier_act.detach().numpy()
        max_act_index = np.where(classifier_num == np.max(classifier_num))
        print(max_act_index[0] + 1)


# #------------------test one picture---------------------------------------
# vgg_face = Vgg_face()
# vgg_face.load_state_dict(torch.load('F:/Code/model/vgg_face_dag.pth'))
# image_path = 'D:/cnnface/femaletrain/n00000068/0026_01.jpg'
# crop_coord = (104, 21, 252, 223)
#
# test_onepicture(vgg_face,image_path,crop_coord)


#------------------test dataSet-------------------------------------------
#load model
vgg_id = Vgg_identity()
vgg_id.load_state_dict(torch.load('F:/Code/model/vgg_identity_CrossEntro.pth'))
# load and transform pictures
images_path = 'D:/cnnface/cannot_test.csv'

model_target,actual_target = test_dataSet(vgg_id,images_path)
