import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PicDataset(Dataset):
    """
    Build a dataset to load pictures
    """
    def __init__(self, csv_file, transform=None, crop=None):
        """
        Initialize PicDataset

        Parameters:
        ------------
        csv_file[str]:  table contains picture names, conditions and picture onset time.
                        This csv_file helps us connect cnn activation to brain images.
                        Please organize your information as:

                        [PICDIR]
                        stimID          condition   onset(optional) measurement(optional)
                        download/face1  face        1.1             3
                        mgh/face2.png   face        3.1             5
                        scene1.png      scene       5.1             4

        transform[callable function]: optional transform to be applied on a sample.
        crop[bool]:crop picture optionally by a bounding box.
                   The coordinates of bounding box for crop pictures should be measurements in csv_file.
                   The label of coordinates in csv_file should be left_coord,upper_coord,right_coord,lower_coord.
        """
        self.csv_file = pd.read_csv(csv_file, skiprows=1)
        with open(csv_file,'r') as f:
            self.picpath = f.readline().rstrip()
        self.transform = transform
        picname = np.array(self.csv_file['stimID'])
        condition = np.array(self.csv_file['condition'])
        self.picname = picname
        self.condition = condition
        self.crop = crop
        if self.crop:
            self.left = np.array(self.csv_file['left_coord'])
            self.upper = np.array(self.csv_file['upper_coord'])
            self.right = np.array(self.csv_file['right_coord'])
            self.lower = np.array(self.csv_file['lower_coord'])

    def __len__(self):
        """
        Return sample size
        """
        return self.csv_file.shape[0]

    def __getitem__(self, idx):
        """
        Get picture name, picture data and target of each sample

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        picname: picture name
        picimg: picture data, save as a pillow instance
        target_label: target of each sample (label)
        """
        # load pictures
        target_name = np.unique(self.condition)
        picimg = Image.open(os.path.join(self.picpath, self.picname[idx])).convert('RGB')
        if self.crop:
            picimg = picimg.crop((self.left[idx],self.upper[idx],self.right[idx],self.lower[idx]))
        target_label = target_name.tolist().index(self.condition[idx])
        if self.transform:
            picimg = self.transform(picimg)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            picimg = self.transform(picimg)
        return picimg, target_label

    def get_picname(self, idx):
        """
        Get picture name and its condition (target condition)

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        picname: picture name
        condition: target condition
        """
        return os.path.basename(self.picname[idx]), self.condition[idx]