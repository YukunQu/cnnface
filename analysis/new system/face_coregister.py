# 这个脚本将原始的面孔刺激配准到中性模板面孔上

#
# 1.diaoyong face morph li de daima
#
# 2. huo de template  de 68 ge zuobiao
#
# 3. 对每个刺激进行warp
#
# 4.生成新的刺激面孔。

from docopt import docopt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from facemorpher import locator
from facemorpher import aligner
from facemorpher import warper
from facemorpher import blender
from facemorpher import plotter

def list_imgpaths(imgfolder):
    for fname in os.listdir(imgfolder):
        if (fname.lower().endswith('.jpg') or
                fname.lower().endswith('.png') or
                fname.lower().endswith('.jpeg')):
            yield os.path.join(imgfolder, fname)

def sharpen(img):
    blured = cv2.GaussianBlur(img, (0, 0), 2.5)
    return cv2.addWeighted(img, 1.4, blured, -0.4, 0)

def load_image_points(path, size):
    img = cv2.imread(path)
    points = locator.face_points(img)

    if len(points) == 0:
        print('No face in %s' % path)
        return None, None
    else:
        return aligner.resize_align(img, points, size)


def face2template(imgpaths, dest_filename=None, width=500, height=600, background='black',
                  blur_edges=False, out_filename='.png',stim_class=None, plot=False):

    size = (height, width)

    images = []
    point_set = []

    if dest_filename is not None:
        dest_img, dest_points = load_image_points(dest_filename, size)
    if dest_img is None or dest_points is None:
        raise Exception('No face or detected face points in dest img: ' + dest_filename)
    num_images = 0
    for i,path in enumerate(imgpaths):
        img, points = load_image_points(path, size)
        if img is not None:
            num_images += 1
            result_image = np.uint8(warper.warp_image(img, points,
                                                      dest_points, size, np.float32))
            out_filename_img = out_filename.format(stim_class,i+1)
            plt = plotter.Plotter(plot, num_images=1, out_filename=out_filename_img)
            plt.save(result_image)
    if num_images == 0:
        raise FileNotFoundError('Could not find any valid images.' +
                                ' Supported formats are .jpg, .png, .jpeg')
    print('Coregister {} images'.format(num_images))


def main():
    stim_class = ['female','male']
    for stim in stim_class:
        imagefolder = r'D:\cnnface\analysis_for_reply_review\data\train\{}'.format(stim)
        imgpaths = list_imgpaths(imagefolder)
        dest_filename = r'D:\cnnface\gender_analysis\face_template\gray/baseface.jpg'
        out_filename = r'D:\cnnface\analysis_for_reply_review\data\registrated\train\{}/{}.png'
        width = 512
        height = 512
        background = 'black'
        blur_edges = True
        plot = False

        face2template(imgpaths, dest_filename=dest_filename, width=width, height=height, background=background,
                 blur_edges=blur_edges, out_filename=out_filename,stim_class=stim,plot=plot)

if __name__ == "__main__":
    main()
