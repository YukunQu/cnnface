import os
import pandas as pd
#extract meta data of special subject from idenity_meta.csv of vggface2

subjectid = os.listdir('D:\cnnface/female_male_test_51_addnoise/pure')

subject_meta = pd.read_csv('D:/VGGface2/meta_data/identity_meta.csv',error_bad_lines=False)

for i,subid in enumerate(subjectid):
    if i == 0:
        subject_meta_part = subject_meta[subject_meta["Class_ID"].str.contains(subid)]
    else:
        subject_meta_tmp = subject_meta[subject_meta["Class_ID"].str.contains(subid)]
        subject_meta_part = pd.concat([subject_meta_part,subject_meta_tmp])

#Count the number of female pictures or male pictures
subject_fe_meta_part = subject_meta_part[subject_meta_part[" Gender"].str.contains(' f')]
subject_ma_meta_part = subject_meta_part[subject_meta_part[" Gender"].str.contains(' m')]

female_id = list(subject_fe_meta_part["Class_ID"])
male_id = list(subject_ma_meta_part["Class_ID"])

prepath = 'D:\cnnface/female_male_test_51_addnoise/pure'
female_path = [prepath +'/'+ id for id in female_id]
male_path = [prepath + '/' + id for id in male_id]

pic_num = [len(os.listdir(path)) for path in female_path]
female_pic_num = sum(pic_num)
pic_num = [len(os.listdir(path)) for path in male_path]
male_pic_num = sum(pic_num)
