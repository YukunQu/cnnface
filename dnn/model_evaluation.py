import numpy as np
import pandas as pd
import sklearn.metrics as sm


# 综合一些评估模型的指标
file_dir = r'D:\cnnface\analysis_for_reply_review\result&figure\result2/vggface'
expect_label_path = file_dir +'/expect_label.npy'
label_path = file_dir +'/label.npy'

expect_label = np.load(expect_label_path)
label = np.load(label_path)

model_report = sm.classification_report(expect_label, label, target_names=["Female", 'Male'], digits=3, output_dict=True)
model_report = pd.DataFrame(model_report).transpose()
#model_report.to_csv(r'D:\cnnface\analysis_for_reply_review\result&figure\result2\alexnet/model_report.csv')

confusion_matrix = sm.confusion_matrix(expect_label, label, labels=[0, 1])