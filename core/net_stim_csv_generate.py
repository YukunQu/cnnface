from dnnbrain.dnn.io import generate_stim_csv
import os
import pandas as pd


def read_Imagefolder(prepath):
    """
    The function read from a already organized Image folder and return picname list and condition list
    for generate csv file more quickly.

    :param prepath[str]:already organized folder
    :return:
        picname[list]:contains all name of Images in prepath
        picpath[list]:contains all subpath of Images in prepath
        condition[list]:contains the class of all Images
    """
    test_set = list(os.walk(prepath))
    picname = []
    picpath = []
    condition = []
    for label in test_set[1:]:
        condition_name = label[0].split('\\')[-1]
        picname_tem = [pic for pic in label[2]]
        picpath_tem = [condition_name + '/' + pic for pic in label[2]]
        condition_tem = [condition_name for i in label[2]]
        picname.append(picname_tem)
        picpath.append(picpath_tem)
        condition.append(condition_tem)

    picname = sum(picname,[])
    picpath = sum(picpath,[])
    condition = sum(condition,[])
    return picname,picpath,condition


def read_boundingbox_from_loosebb(subjectid,picname):
    """read coordinates of bounding box from loose_bb_train.csv.

       subjectid[list]:contains subject id in vggface2
       picpath_list[list]:contains the name of the image that you want to get the coordinate.


    """
    boundingbox = pd.read_csv('D:/VGGface2/meta_data/bb_landmark/loose_bb_train.csv')


    for i,picname in enumerate(subjectid):
        if i==0:
            sub_boundingbox = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
        else:
            sub_boundingbox_suffix = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
            sub_boundingbox = pd.concat([sub_boundingbox,sub_boundingbox_suffix])

    name_id = [name.split('.')[0] for name in picname]
    for i,picname in enumerate(name_id):
        if i==0:
            pic_boundingbox = sub_boundingbox[sub_boundingbox["NAME_ID"].str.contains(picname)]
        else:
            pic_boundingbox_suffix = sub_boundingbox[sub_boundingbox["NAME_ID"].str.contains(picname)]
            pic_boundingbox = pd.concat([pic_boundingbox,pic_boundingbox_suffix])

    #generate coordinate list.
    left_coord = pic_boundingbox["X"].tolist()
    upper_coord = pic_boundingbox["Y"].tolist()
    right_coord = [left_coord[i]+pic_boundingbox["W"].tolist()[i] for i in range(len(left_coord))]
    lower_coord = [upper_coord[i]+pic_boundingbox["H"].tolist()[i] for i in range(len(left_coord))]
    beh_measure = {'left_coord':left_coord,'upper_coord':upper_coord,'right_coord':right_coord,'lower_coord':lower_coord}
    return beh_measure


prepath = 'D:\cnnface/female_male_test_51_addnoise/Face_template/gabor_noise'
output = 'D:\cnnface/female_male_test_51_addnoise/rcicr_gabor_noise.csv'
picname,picpath,condition = read_Imagefolder(prepath)
#subjectid = os.listdir(prepath)
#beh_measure = read_boundingbox_from_loosebb(subjectid, picpath)
#,behavior_measure=beh_measure
generate_stim_csv(prepath,picpath,condition,output)
