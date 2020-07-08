from facemorpher import averager, morpher,list_imgpaths

#%% face averager

# def averager(imgpaths, dest_filename=None, width=500, height=600, background='black',
#             blur_edges=False, out_filename='result.png', plot=False):
imagefolder = r'D:\cnnface\facegen\male'
imgpaths = list_imgpaths(imagefolder)
out_filename = r'D:\AI_generate_faces\male/male_average.png'
width = 512
height = 512
background = 'black'
blur_edges = True
plot = False

averager(imgpaths, dest_filename=None, width=width, height=height, background=background,
            blur_edges=blur_edges, out_filename=out_filename, plot=plot)


#%% face morph
imagefolder = r'D:\AI_generate_faces\gray\en'
imgpaths = list_imgpaths(imagefolder)

#def morpher(imgpaths, width=500, height=600, num_frames=20, fps=10,
# out_frames=None, out_video=None, plot=False, background='black'):
width = 512
height = 512
num_frames = 502
fps = 10
out_frames = r'D:\AI_generate_faces\morphface'
out_video = None

morpher(imgpaths, width=width, height=height, num_frames=num_frames, fps=fps,
        out_frames=out_frames, out_video=out_video, plot=False, background='black')

