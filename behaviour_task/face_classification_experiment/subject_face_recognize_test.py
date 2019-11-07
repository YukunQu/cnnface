import os
import numpy as np
import pandas as pd
from psychopy import visual, core, event, gui
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  # alculate the false alarm and hit rate
    roc_auc = auc(fpr, tpr)  # calcuate the auc

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  # false alarm为横坐标，hit rate为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return tpr, fpr, threshold, roc_auc

# load images path
csv_file = r'D:\cnnface\female_male_test_51_addnoise/Face_template.csv'
images = pd.read_csv(csv_file, skiprows=1)
with open(csv_file, 'r') as f:
    prepath = f.readline().rstrip()
picNames = np.array(images['stimID'])
images_path = [os.path.join(prepath, picName) for picName in picNames]

# check the category directory exist
prepath = ''
task = ['gender', 'Emotion', 'Identity', 'train']
for i in task:
    test_path = os.path.join(prepath, i)
    if os.path.exists(test_path) is False:
        os.mkdir(test_path)

# subject information recorded
info = {'姓名': '', '年龄': '', '性别': '', 'task': ['gender', 'Emotion', 'Identity', 'train'], 'part': [1, 2]}
infoSub = gui.DlgFromDict(dictionary=info, title='基本信息',order=['姓名', '年龄', 'task', 'part'])
if infoSub.OK == False:
    core.quit()

saveprepath = os.path.join(prepath, info['task'])
fileSavePath = '{}.csv'.format((saveprepath + '/' + info['task'] + '_' + info['part'] + '_' + info['姓名']))
with open(fileSavePath, 'a') as f:
    f.write(info['姓名']+','+info['年龄']+','+info['性别']+','+info['task']+','+info['part']+'\n')

# material prepare
win = visual.Window(size=(1600, 1200), color=(0, 0, 0), fullscr=False,gammaErrorPolicy='ignore')
rscale = visual.RatingScale(win, choices=['很可能是女性', '也许是女性', '也许是男性', '很可能是男性'], textSize=0.45, pos=[0, -0.6],
                             markerColor='DarkRed', singleClick=True, minTime=0.6)

pre_text_1 = '实验即将开始，请判断接下来每一张面孔是男性还是女性'
text_1 = visual.TextStim(win, text =pre_text_1, pos=(0, 0), color=(-1, -1, -1), height=0.07, bold=True)
pre_text_2 = '如果准备好，请按任意键开始'
text_2 = visual.TextStim(win, text =pre_text_2, pos=(0, -0.2), color=(-1, -1, -1), height=0.07, bold=True)
pic = visual.ImageStim(win)
fixation = visual.Circle(win,radius=0.01,size=[3,4], fillColor='white')

ratings = []
RTs = []
# -----------------------------Experiment_start--------------------------------------------------------
text_1.draw()
text_2.draw()
win.flip()
key = event.waitKeys(maxWait=10)

for img in images_path:
    pic.image = img
    pic.draw()
    win.flip()
    core.wait(1)
    rscale.reset()
    timer = core.CountdownTimer(2)
    while timer.getTime() > 0:
        rscale.draw()
        win.flip()
    rating = rscale.getRating()
    print(type(rating))
    if rating == '很可能是女性':
        rating = 1
    elif rating == '很可能是男性':
        rating = -1
    elif rating == '也许是女性':
        rating = 0.1
    else:
        rating = -0.1

    decisionTime = rscale.getRT()
    ratings.append(rating)
    RTs.append(decisionTime)
    fixation.draw()
    win.flip()
    core.wait(0.6)

win.close()
# save subject information and experiment information

if info['task'] == 'train':
    actual_label = ratings
    target_label = [0]*50 + [1] * 50
    tpr, fpr, threshold, roc_auc = acu_curve(actual_label, target_label)
    print(tpr, fpr, threshold, roc_auc)

df = pd.DataFrame({'Rating': ratings, 'RTs': RTs})
df.to_csv(fileSavePath, mode='a')
