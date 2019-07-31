import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset,DataLoader
from dnnbrain.dnn.models import Vgg_face,dnn_test_model


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
        classifier_act = vgg_face(image)
        classifier_act = classifier_act.squeeze(0)
        classifier_num = classifier_act.detach().numpy()
        max_act_index = np.where(classifier_num == np.max(classifier_num))
        print(max_act_index[0] + 1)



# #------------------test one picture---------------------------------------
vgg_face = Vgg_face()
vgg_face.load_state_dict(torch.load('F:/Code/model/vgg_face_dag.pth'))
image_path = 'D:/cnnface/femaletrain/n00000068/0026_01.jpg'
crop_coord = (104, 21, 252, 223)

test_onepicture(vgg_face,image_path,crop_coord)


#------------------test dataSet-------------------------------------------
#load model
vgg_face = Vgg_face()
vgg_face.load_state_dict(torch.load('F:/Code/model/vgg_face_dag.pth'))
# load and transform pictures
images_path = 'D:/cnnface/female_crossEntropLoss_train.csv'

model_target,actual_target = test_dataSet(vgg_face,images_path)

actual_target  = [67 if i == 0 else 1585 for i in actual_target]
model_target = np.array(model_target)
actual_target = np.array(actual_target)
test_acc = 1.0 * np.sum(model_target == actual_target) / len(actual_target)
print(test_acc)
acc_67 = model_target.tolist().count(67)/actual_target.tolist().count(67)
acc_1585 = model_target.tolist().count(1585)/actual_target.tolist().count(1585)
print(acc_67,acc_1585)