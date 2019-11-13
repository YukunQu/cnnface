import os
import numpy as np
import pandas as pd
from psychopy import visual, core, event, gui

# set path of experiment
prePath = r'D:\Exp\faceRepresentationExp\exp'

# subject information recorded and task selected
info = {'姓名': '', '年龄': '', '性别': ['男','女'], 'task': ['gender', 'Emotion', 'Identity'], 'part': [1, 2, 3, 4, 5]}
infoSub = gui.DlgFromDict(dictionary=info, title='基本信息', order=['姓名', '年龄', '性别','task', 'part'])
if infoSub.OK == False:
    core.quit()

# load images path
csv_file =  os.path.join(prePath,'stimulus',info['task'],'part{}.csv'.format(info['part']))
print(csv_file)
images = pd.read_csv(csv_file, skiprows=1)
with open(csv_file, 'r') as f:
    prepath = f.readline().rstrip()
picNames = np.array(images['stimID'])
images_path = [os.path.join(prepath, picName) for picName in picNames]

# write the subject information to csv file
savefilename = '{}.csv'.format(info['task'] + '_' + info['part'] + '_' + info['姓名'])
fileSavePath = os.path.join(prePath, 'result', info['task'], 'part'+info['part'], savefilename)
with open(fileSavePath, 'a') as f:
    f.write(info['姓名']+','+info['年龄']+','+ info['性别']+','+info['task']+','+info['part']+'\n')

# material prepare  
event.globalKeys.add(key='q', func=core.quit, name ='shutdown')

win = visual.Window(size=(1920, 1080), color=(0, 0, 0), fullscr=False, gammaErrorPolicy='ignore')
rscale = visual.RatingScale(win, choices=['很可能是女性', '也许是女性', '也许是男性', '很可能是男性'], textSize=0.6, pos=[0, -0.6],
                             markerColor='DarkRed', singleClick=True, minTime=0,respKeys=['z','x','n','m'],size=1.3,
                             markerStart=-10,noMouse=True)

pre_text_1 = '实验即将开始，请判断接下来每一张面孔是男性还是女性'
text_1 = visual.TextStim(win, text=pre_text_1, pos=(0, 0), color=(-1, -1, -1), height=0.07, bold=True)
pre_text_2 = '如果准备好，请按任意键开始'
text_2 = visual.TextStim(win, text=pre_text_2, pos=(0, -0.2), color=(-1, -1, -1), height=0.07, bold=True)
relax_text = '休息时间'
relax = visual.TextStim(win, text=relax_text, pos=(0, 0), color=(-1, -1, -1), height=0.07, bold=True)
next_stage_text = '下一阶段实验即将开始，请准备'
next_stage = visual.TextStim(win, text=next_stage_text, pos=(0, 0), color=(-1, -1, -1), height=0.07, bold=True)
pic = visual.ImageStim(win)
ratings = []
RTs = []

# -----------------------------Experiment_start-------------------------------------------------------- #
text_1.draw()
text_2.draw()
win.flip()
key = event.waitKeys(maxWait=10)

for i,img in enumerate(images_path):
    if i % 100 == 0 and i != 0 :
        relax.draw()
        win.flip()
        core.wait(60)
        next_stage.draw()
        win.flip()
        core.wait(5)
    pic.image = img
    pic.draw()
    win.flip()
    core.wait(1)
    rscale.reset()
    while rscale.noResponse:
        rscale.draw()
        win.flip()
    core.wait(0.4)
    win.flip()
    core.wait(0.6)
    rating = rscale.getRating()
    print(rating)
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

# save subject information and experiment information
df = pd.DataFrame({'Rating': ratings, 'RTs': RTs})
df.to_csv(fileSavePath, mode='a')