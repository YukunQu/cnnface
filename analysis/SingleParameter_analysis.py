# ------------------------------------------------------------------------------ #
# Check how the single parameter effect the classification probability of vgg    #
# ------------------------------------------------------------------------------ #
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from dnnbrain.dnn.io import PicDataset, DataLoader
from cnnface.dnn.vgg_identity_recons import Vgg_identity

# Get the base classification probability of baseFace



# calculate the probability result of baseface
