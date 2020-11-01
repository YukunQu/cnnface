import os
import shutil
import pandas as pd

# extract the top50 id folder to two stream stimuli dataset.
id_top1000 = pd.read_csv(r'F:\vggface2/identity_top1000_meta.csv')
dir_names = id_top1000['Class_ID']


src_dir = r'F:\vggface2\stimuli\train'
des_dir = r'F:\vggface2/train'

for dir in dir_names[1:]:
    src_path = os.path.join(src_dir,dir)
    des_path = os.path.join(des_dir,dir)
    shutil.move(src_path, des_path)
