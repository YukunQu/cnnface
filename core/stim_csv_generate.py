from dnnbrain.dnn.io import generate_stim_csv
import os
import pandas as pd


def generate_vggface2_stim_csv(prepath,output):
    """
    The function for generating a stimulus csv file with special structure
    from a already organized folder and vggface2 loose_bb_train file.
    :param prepath[str]:already organized folder
    :param output[str]:the output path of stimulus csv file
    :return:
    """

    # Get picname list and condition list form the floder.
    test_set = list(os.walk(prepath))
    picname_list = []
    condition_list = []
    for label in test_set[1:]:
        condition = label[0].split('\\')[-1]
        picname_path = [condition + '/' + pic for pic in label[2]]
        picname_list.append(picname_path)
        condition = [condition for i in range(len(label[2]))]
        condition_list.append(condition)
    picname_list = sum(picname_list,[])
    condition_list = sum(condition_list,[])

    #read coordinates of bounding box from loose_bb_train.csv.
    boundingbox = pd.read_csv('D:/VGGface2/meta_data/bb_landmark/loose_bb_train.csv')

    subjectid = ['n001506','n005591']
    for i,picname in enumerate(subjectid):
        if i==0:
            sub_boundingbox = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
        else:
            sub_boundingbox_suffix = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
            sub_boundingbox = pd.concat([sub_boundingbox,sub_boundingbox_suffix])

    name_id = [name.split('.')[0] for name in picname_list]
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

    #generate stimulus_csv file.
    generate_stim_csv(prepath,picname_list,condition_list,output,behavior_measure=beh_measure)


prepath = 'D:/cnnface/femaletrain'
output = 'D:/cnnface/female_crossEntropLoss_train.csv'
generate_vggface2_stim_csv(prepath,output)