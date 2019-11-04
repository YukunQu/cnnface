import numpy as np
from psychopy import visual, core, event


# load images

images_path = []
ratings = []
RTs = []

# material prepare
win = visual.Window(size = (1600,1200), color = (0,0,0), fullscr=False,gammaErrorPolicy='ignore')
rscale = visual.RatingScale(win, choices = ['很可能是女性', '也许是女性', '也许是男性', '很可能是男性'], textSize = 0.45, pos=[0, -0.6],
                             markerColor='DarkRed',singleClick=True, minTime=0.6)


pre_text_1 = '实验即将开始，请判断接下来每一张面孔是男性还是女性'
text_1 = visual.TextStim(win, text =pre_text_1, pos=(0,0), color=(-1,-1,-1),height=0.07, bold=True)
pre_text_2 = '如果准备好，请按任意键开始'
text_2 = visual.TextStim(win, text =pre_text_2, pos=(0,-0.2), color=(-1,-1,-1),height=0.07, bold=True)
pic = visual.ImageStim(win, image=r'F:\rcicr\RcicrPsychopyExample\RcicrPsychoPyExample/MNES.jpg')

# -----------------------------Experiment_start--------------------------------------------------------
text_1.draw()
text_2.draw()
win.flip()
key = event.waitKeys(maxWait=10)


for img in images_path:
    pic.image = img
    rscale.reset()
    while rscale.noResponse:
        pic.draw()
        rscale.draw()
        win.flip()

    rating = rscale.getRating()
    decisionTime = rscale.getRT()
    ratings.append(rating)
    RTs.append(decisionTime)

win.close()