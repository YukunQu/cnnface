from psychopy import visual, core, event #import some libraries from PsychoPy

#create a window
win = visual.Window([800,600],monitor="testMonitor", units="deg",gammaErrorPolicy='ignore')

ratingScale = visual.RatingScale(
    win, choices=['agree', 'disagree'],

    markerStart=0.5, markerColor='DarkRed',
    singleClick=True,minTime=0.6)

while ratingScale.noResponse:
    ratingScale.draw()
    win.flip()

win.close()