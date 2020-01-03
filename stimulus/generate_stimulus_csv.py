from dnnbrain.dnn.io import generate_stim_csv
import os
import pandas as pd


def read_Imagefolder(parpath):
    """
    The function read from a already organized Image folder and return picname list and condition list
    for generate csv file more quickly.

    :param prepath[str]:already organized folder
    :return:
        picname[list]:contains all name of Images in prepath
        picpath[list]:contains all subpath of Images in prepath
        condition[list]:contains the class of all Images
    """
    test_set = list(os.walk(parpath))

    picpath = []
    condition = []
    if len(test_set) == 1:  # if the folder only have pictures, the folder name will be the condition
        label = test_set[0]
        condition_name = os.path.basename(label[0])
        picpath_tem = label[2]
        condition_tem = [condition_name for i in label[2]]
        picpath.append(picpath_tem)
        condition.append(condition_tem)
    else:                   # if the folder have have some sub-folders, the sub-folders name will be the conditions
        for label in test_set[1:]:
            condition_name = os.path.basename(label[0])
            picpath_tem = [condition_name + '/' + pic for pic in label[2]]
            condition_tem = [condition_name for i in label[2]]
            picpath.append(picpath_tem)
            condition.append(condition_tem)

    picpath = sum(picpath,[])
    condition = sum(condition,[])
    return picpath, condition


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
            #sub_boundingbox[sub_boundingbox["NAME_ID"].isin(name_id)]
            # 还有一个问题，虽然取出了name_id存在的列，但是顺序并不一定是跟name_id一致的。这个程序后面还得跑一遍，但速度会快很多。
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


if __name__ == '__main__':
    prepath = r'D:\VGGface2\overlap_vggface1_2\n004029'
    output = r'D:\VGGface2\overlap_vggface1_2\n004029.csv'
    picpath,condition = read_Imagefolder(prepath)
    # subjectid = os.listdir(prepath)
    # picname = [os.path.base(picpath) for p in picpath]
    # beh_measure = read_boundingbox_from_loosebb(subjectid, picname)
    # ,behavior_measure=beh_measure
    generate_stim_csv(prepath, picpath, condition, output)