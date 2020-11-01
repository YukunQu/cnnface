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


def read_boundingbox_from_loosebb(subjectid, picname):
    """
    read coordinates of bounding box from loose_bb_train.csv.

    subjectid[list]:contains subject id in vggface2
    picpath_list[list]:contains the name of the image that you want to get the coordinate.

    """
    boundingbox = pd.read_csv('D:/VGGface2/meta_data/bb_landmark/loose_bb_train.csv')


    for i, picname in enumerate(subjectid):
        if i == 0:
            sub_boundingbox = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
        else:
            sub_boundingbox_suffix = boundingbox[boundingbox["NAME_ID"].str.contains(picname)]
            sub_boundingbox = pd.concat([sub_boundingbox,sub_boundingbox_suffix])

    name_id = [name.split('.')[0] for name in picname]
    for i,picname in enumerate(name_id):
        if i==0:
            pic_boundingbox = sub_boundingbox[sub_boundingbox["NAME_ID"].str.contains(picname)]
            # sub_boundingbox[sub_boundingbox["NAME_ID"].isin(name_id)]
            # 还有一个问题，虽然取出了name_id存在的列，但是顺序并不一定是跟name_id一致的。这个程序后面还得跑一遍，但速度会快很多。
        else:
            pic_boundingbox_suffix = sub_boundingbox[sub_boundingbox["NAME_ID"].str.contains(picname)]
            pic_boundingbox = pd.concat([pic_boundingbox,pic_boundingbox_suffix])

    #generate coordinate list.
    left_coord = pic_boundingbox["X"].tolist()
    upper_coord = pic_boundingbox["Y"].tolist()
    right_coord = [left_coord[i]+pic_boundingbox["W"].tolist()[i] for i in range(len(left_coord))]
    lower_coord = [upper_coord[i]+pic_boundingbox["H"].tolist()[i] for i in range(len(left_coord))]
    coord_box = {'left_coord':left_coord,'upper_coord':upper_coord,'right_coord':right_coord,'lower_coord':lower_coord}
    return coord_box

def read_boundingbox(picpath):
    boundingbox = pd.read_csv('D:/VGGface2/meta_data/bb_landmark/loose_bb_train.csv')
    name_id = [name.split('.')[0] for name in picpath]

    idx = []
    for i, name in enumerate(boundingbox["NAME_ID"]):
        if name in name_id:
            idx.append(i)
        if i%1000 ==0:
            print(i)
    sub_boundingbox = boundingbox.iloc[idx,:]

    left_coord = sub_boundingbox["X"].tolist()
    upper_coord = sub_boundingbox["Y"].tolist()
    right_coord = [left_coord[i]+sub_boundingbox["W"].tolist()[i] for i in range(len(left_coord))]
    lower_coord = [upper_coord[i]+sub_boundingbox["H"].tolist()[i] for i in range(len(left_coord))]
    coord_box = {'left_coord':left_coord,'upper_coord':upper_coord,'right_coord':right_coord,'lower_coord':lower_coord}
    return coord_box


def generate_stim_csv(parpath, picname_list, condition_list, outpath, onset_list=None, behavior_measure=None):
    """
    Automatically generate stimuli table file.
    Noted that the stimuli table file satisfied follwing structure and sequence needs to be consistent:

    [PICDIR]
    stimID              condition   onset(optional) measurement(optional)
    download/face1.png  face        1.1             3
    mgh/face2.png       face        3.1             5
    scene1.png          scene       5.1             4

    Parameters:
    ------------
    parpath[str]: parent path contains stimuli pictures
    picname_list[list]: picture name list, each element is a relative path (string) of a picture
    condition_list[list]: condition list
    outpath[str]: output path
    onset_list[list]: onset time list
    behavior_measure[dictionary]: behavior measurement dictionary
    """
    assert len(picname_list) == len(condition_list), 'length of picture name list must be equal to condition list.'
    assert os.path.basename(outpath).endswith('csv'), 'Suffix of outpath should be .csv'
    picnum = len(picname_list)
    if onset_list is not None:
        onset_list = [str(ol) for ol in onset_list]
    if behavior_measure is not None:
        list_int2str = lambda v: [str(i) for i in v]
        behavior_measure = {k:list_int2str(v) for k, v in behavior_measure.items()}
    with open(outpath, 'w') as f:
        # First line, parent path
        f.write(parpath+'\n')
        # Second line, key names
        table_keys = 'stimID,condition'
        if onset_list is not None:
            table_keys += ','
            table_keys += 'onset'
        if behavior_measure is not None:
            table_keys += ','
            table_keys += ','.join(behavior_measure.keys())
        f.write(table_keys+'\n')
        # Three+ lines, Data
        for i in range(picnum):
            data = picname_list[i]+','+condition_list[i]
            if onset_list is not None:
                data += ','
                data += onset_list[i]
            if behavior_measure is not None:
                for bm_keys in behavior_measure.keys():
                    data += ','
                    data += behavior_measure[bm_keys][i]
            f.write(data+'\n')


if __name__ == '__main__':
    prepath = r'D:\cnnface\analysis_for_reply_review\data\registrated\train'
    output = r'D:\cnnface\analysis_for_reply_review\data\registrated\train.csv'

    picpath, condition = read_Imagefolder(prepath)

    boundbox = False
    if boundbox == True:
        coord_box = read_boundingbox(picpath)
        generate_stim_csv(prepath,picpath, condition, output, behavior_measure=coord_box)
    else:
        generate_stim_csv(prepath, picpath, condition, output)