import os
import numpy as np
from PIL import Image
from cnnface.analysis.ci_analysis import ClassImage


class PrototypeFace:
    def __init__(self,baseface,ci):
        self.baseface = baseface
        self.ci = ci

    def load_baseface(self, baseface_path):
        self.baseface = Image.open(baseface_path)

    def recon_face(self,scale,ci=None):
        """
     Reconstruct face from baseface and ci

     Parameter:
     ---------------------------------------------------------------------------------------------
     baseface[PIL]: PIL file contains the baseface
     ci[array]: 2D array, contains the classification image which can be generated from generateCI

     Return:
     ---------------------------------------------------------------------------------------------
     img_add,img_sub[PIL]: reconstruct face
     """
        baseface_img = np.array(self.baseface).astype('float64')

        if ci is not None:
            ci_scaled = ci * scale
        else:
            ci_scaled = self.ci * scale
        bf_add = baseface_img + ci_scaled
        bf_sub = baseface_img - ci_scaled

        bf_add[bf_add > 255] = 255
        bf_add[bf_add < 0] = 0
        bf_sub[bf_sub > 255] = 255
        bf_sub[bf_sub < 0] = 0

        img_add = Image.fromarray(bf_add.astype('int8')).convert('L')
        img_sub = Image.fromarray(bf_sub.astype('int8')).convert('L')
        return img_add, img_sub

    def recon_faces(self,savepath=None):
        scales = [2, 4, 8, 16, 32]
        faces_add = []
        faces_sub = []
        # generate the reconstruct face in different frequency scales
        for scale, img in zip(scales,self.ci):
            scaleIndex = 45/img.max()
            img_add, img_sub = self.recon_face(ci=img, scale=scaleIndex)
            faces_add.append(np.array(img_add))
            faces_sub.append(np.array(img_sub))

            if isinstance(savepath,str):

                picSavePath_add = r'{}/add/face_{}_add.jpg'.format(savepath,scale)
                picSavePath_sub = r'{}/sub/face_{}_sub.jpg'.format(savepath,scale)
                img_add.save(picSavePath_add,quality=95)
                img_sub.save(picSavePath_sub,quality=95)
        return faces_add, faces_sub

#
# baseface = Image.open(r'D:\cnnface\Data_sorted\commonData/baseface.jpg')
# cis = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/cis_vggface_activaiton.npy')
# vgg_prototype = PrototypeFace(baseface,cis)
# save_path = r'D:\cnnface\Data_sorted\vggface\prototype_face\differentScale'
# vgg_prototype.recon_faces(save_path)
