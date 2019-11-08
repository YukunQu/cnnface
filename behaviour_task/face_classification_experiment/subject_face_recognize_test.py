import os
import numpy as np
import pandas as pd
from psychopy import visual, core, event, gui


# subject information recorded and task selected
info = {'姓名': '', '年龄': '', '性别': '', 'task': ['gender', 'Emotion', 'Identity', 'train'], 'part': [1, 2]}
infoSub = gui.DlgFromDict(dictionary=info, title='基本信息', order=['姓名', '年龄', 'task', 'part'])
if infoSub.OK == False:
    core.quit()

# load images path
csv_file = r'./{}.csv'.format(info['task'])
images = pd.read_csv(csv_file, skiprows=1)
with open(csv_file, 'r') as f:
    prepath = f.readline().rstrip()
picNames = np.array(images['stimID'])
images_path = [os.path.join(prepath, picName) for picName in picNames]

# write the subject information to csv file
saveprepath = os.path.join('./', info['task'])
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
ratings = []
RTs = []

# -----------------------------Experiment_start-------------------------------------------------------- #
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
    timer = core.CountdownTimer(2.5)
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
win.close()
# save subject information and experiment information

if info['task'] == 'train':
    actual_label = np.array(ratings)
    actual_label[actual_label > 0] = 1
    actual_label[actual_label < 0] = 0
    target_label = np.squeeze(np.array([0]*50 + [1] * 50))
    hitRate = len(np.where(actual_label[:2]>0)) / 4
    falsealarm = len(np.where(actual_label[2:4] < 0)) / 4
    dprime=''
    print(hitRate, falsealarm,dprime)

df = pd.DataFrame({'Rating': ratings, 'RTs': RTs})
df.to_csv(fileSavePath, mode='a')