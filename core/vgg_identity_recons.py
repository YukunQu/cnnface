import torch
import torch.nn as nn
from dnnbrain.dnn.models import Vgg_face


def vgg_face(weights_path=None, **kwargs):
    """
    load imported model instance
    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model


class Vgg_identity(Vgg_face):
    def __init__(self):
        super().__init__()
        self.fc8 = nn.Linear(4096,2)


#Create and save a new classifier (two class)
model = vgg_face('F:/Code/model/vgg_face_dag.pth')

for param in model.parameters():
    param.requires_grad = False
in_features = model.fc8.in_features
out_features = 2
new_fc8 = nn.Linear(in_features,out_features,bias = True)
model.fc8 = new_fc8
print(model)

torch.save(model.state_dict(),'F:/Code/model/vgg_identity_ori.pth')
