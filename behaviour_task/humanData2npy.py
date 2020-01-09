import numpy as np
import pandas as pd
import os


def hum_csv2label(csvfile_path, param_sum):
    sub_suffix = os.path.basename(csvfile_path)
    sub_part = int(sub_suffix.split('_')[1])
    humanData = pd.read_csv(csvfile_path, skiprows=1)
    rating = humanData['Rating']
    label_sub = np.array([1 if r == -1 or r == -0.1 else 0 for r in rating])
    param_sub = param_sum[(sub_part-1) * 1000:sub_part * 1000,:]
    return label_sub, param_sub


if __name__ == '__main__':
    param_sum = np.load(r'D:\cnnface\gender_analysis\noise_stimulus\metadata/params_20000.npy')
    prepath = r'D:\cnnface\gender_analysis\human_result\exp\gender/part{}'
    label_sum = []
    param_exp = []
    names = []
    for i in range(1,12):
        part_path = prepath.format(i)
        part_sub = os.listdir(part_path)
        for sub in part_sub:
            csv_file_path = os.path.join(part_path, sub)
            print(sub)
            print(csv_file_path)
            label_sub, param_sub = hum_csv2label(csv_file_path, param_sum)
            names.extend(sub)
            label_sum.extend(label_sub)
            param_exp.extend(param_sub)
    label_sum = np.array(label_sum)
    param_exp = np.array(param_exp)
    np.save(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/subject_name.npy', names)
    np.save(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/label_sum.npy', label_sum)
    np.save(r'D:\cnnface\gender_analysis\human_result\exp\gender\label/param_exp.npy', param_exp)
