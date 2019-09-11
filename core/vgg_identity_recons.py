import torch
import torch.nn as nn
from dnnbrain.dnn.models import Vgg_face


#constrcture vgg_identity model
class Vgg_identity(Vgg_face):
    def __init__(self):
        super().__init__()
        for param in self.parameters():
            param.requires_grad = False
        self.fc8 = nn.Linear(4096,2)


#Create and save a new classifier parameters(two class)
vgg_face = Vgg_face()
vgg_face.load_state_dict(torch.load('F:/Code/pretrained_model/vgg_face_dag.pth'))


in_features = vgg_face.fc8.in_features
out_features = 2
new_fc8 = nn.Linear(in_features,out_features,bias = True)
vgg_face.fc8 = new_fc8

torch.save(vgg_face.state_dict(),'F:/Code/pretrained_model/vgg_identity_ori.pth')





