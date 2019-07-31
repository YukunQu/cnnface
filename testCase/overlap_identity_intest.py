import os
import numpy as np
import pandas as pd

test_name = pd.read_table('D:/VGGface2/data/test_list.txt',sep='/')
overlap_name = pd.read_table('D:/VGGface2/meta_data/class_overlap_vgg1_2.txt',sep=' ')

test_name = test_name.index.values
overlap_name = overlap_name['VGGFace2'].values

ava_name = []
for i in overlap_name:
    if i in test_name:
        ava_name.append(i)

print(ava_name)