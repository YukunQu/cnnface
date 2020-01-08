import os
import random
import numpy as np
import pandas as pd
from psychopy import visual, core, event, gui


# set path of experiment
prePath = r'F:\exp\faceinnoise\test'

# subject information recorded and task selected
info = {'姓名': '', '年龄': '', '性别': ['女', '男'], 'task': ['gender', 'Emotion', 'Identity']}
infoSub = gui.DlgFromDict(dictionary=info, title='基本信息', order=['姓名', '年龄', '性别', 'task'])
if infoSub.OK == False:
    core.quit()

# load images path
imgs_list = os.listdir(os.path.join(prePath, 'stimuli', info['task']))
picNames = imgs_list
random.shuffle(picNames)
images_path = [os.path.join(prePath, 'stimuli', info['task'], picName) for picName in picNames]

# write the subject information to csv file
savefilename = '{}.csv'.format(info['task'] + '_' + info['姓名'])
fileSavePath = os.path.join(prePath, 'result', info['task'], savefilename)
with open(fileSavePath, 'a') as f:
    f.write(info['姓名']+','+info['年龄']+','+info['性别']+','+info['task']+',')

# material prepare
event.globalKeys.add(key='q', func=core.quit, name ='shutdown')

win = visual.Window(size=(1920, 1080), color=(0, 0, 0), fullscr=False, gammaErrorPolicy='ignore')
rscale = visual.RatingScale(win, choices=['很可能是女性', '也许是女性', '也许是男性', '很可能是男性'],
                            textSize=0.6, pos=[0, -0.6], markerColor='DarkRed', singleClick=True, minTime=0,
                            respKeys=['z','x','n','m'],size=1.3,noMouse=True,markerStart=-10)

pre_text_1 = '实验即将开始，请判断接下来每一张面孔是男性还是女性'
text_1 = visual.TextStim(win, text=pre_text_1, pos=(0, 0), color=(-1, -1, -1), height=0.07, bold=True)
pre_text_2 = '如果准备好，请按任意键开始'
text_2 = visual.TextStim(win, text=pre_text_2, pos=(0, -0.2), color=(-1, -1, -1), height=0.07, bold=True)
pic = visual.ImageStim(win, size=(1,1.77))
# 1
ratings = []
RTs = []
target_label = []
# -----------------------------Experiment_start-------------------------------------------------------- #
text_1.draw()
text_2.draw()
win.flip()
key = event.waitKeys(maxWait=10)

for img in images_path:
    presuffix = os.path.basename(img)
    presuffix = presuffix.split('_')[0]
    print(presuffix)
    if presuffix == 'female':
        target = 1
    elif presuffix == 'male':
        target = 0
    target_label.append(target)
    pic.image = img
    pic.draw()
    win.flip()
    core.wait(60)
    rscale.reset()
    while rscale.noResponse:
        rscale.draw()
        win.flip()
    win.flip()
    core.wait(0.6)
    rating = rscale.getRating()
    if rating == '很可能是女性':
        rating = 1
    elif rating == '很可能是男性':
        rating = -1
    elif rating == '也许是女性':
        rating = 0.1
    elif rating == '也许是男性':
        rating = -0.1
    else:
        rating = 0

    decisionTime = rscale.getRT()
    ratings.append(rating)
    RTs.append(decisionTime)
win.close()

# calculate the hit rate, the false alarm and auc
actual_label = np.array(ratings)
actual_label[actual_label > 0] = 1
actual_label[actual_label < 0] = 0
y_score = np.squeeze(np.array(target_label))

correctNum = [1 if al==ys else 0 for al,ys in zip(actual_label,y_score)]
correctRate = sum(correctNum) / len(y_score)
print('correctRate',correctRate)

# save subject information and experiment information
with open(fileSavePath, 'a') as f:
    f.write(str(correctRate) + '\n')
df = pd.DataFrame({'Rating': ratings, 'targetLabel': y_score,'RTs': RTs})
df.to_csv(fileSavePath, mode='a')
