import cv2
from os.path import join as pjoin
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from dnnbrain.dnn import analyzer
from dnnbrain.dnn import io as dnnio
from scipy import stats
from statsmodels.tsa.tsatools import detrend
import cifti
from ATT.algorithm import tools
from ATT.iofunc import iofiles
from nltk.corpus import wordnet as wn
import pandas as pd
from torchvision import datasets
import matplotlib.pyplot as plt

# prepare DCNN models
def prepare_cnn(modelpara, outunit=1000):
    """
    """
    print('Prepare CNN')
    print('modelparam is {}'.format(modelpara.split('/')[-1]))
    print('outunit is {}'.format(outunit))
    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier[-1] = torch.nn.Linear(4096,outunit)
    alexnet.load_state_dict(torch.load(modelpara, map_location=torch.device('cpu')))
    alexnet.eval()
    return alexnet

# prepare video
def extract_video_frames(cnnmodel, video_name='seg1'):
    """
    """
    print('Prepare Video Frames')
    parpath = '/nfs/e3/natural_vision/liuzm_movie/stimuli/movie'
    # video_name = ['seg'+str(i)+'.mp4' for i in range(1,19,1)]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    cnnmodel = cnnmodel.to(device) 
    print('now execute video {}'.format(video_name+'.mp4'))
    vidcap = cv2.VideoCapture(pjoin(parpath, video_name+'.mp4'))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    output_act = []
    for i in range(framecount):
        if (i+1)%1000 == 0:
            print('...Frame count {}'.format(i+1))
        ifpic, picimg = vidcap.read()
        if ifpic:
            picimg = Image.fromarray(cv2.cvtColor(picimg, cv2.COLOR_BGR2RGB))
            picimg = transform(picimg)
            picimg = picimg.to(device)
            output_tmp = cnnmodel(picimg[None, ...])
            output_act.extend(output_tmp.data.numpy())
    vidcap.release()
    return np.array(output_act), fps

def get_cnn_hrf_signal(actdata, tr=2, fps=30.0, ops=100):
    """
    Remove the first volume and the last 4 volumes
    """
    print('Prepare simulated fMRI')
    actdata = actdata[:14400,:]
    # Data started from 00:16 to 08:16
    # Considering total time started from 00:00 - 08:24,
    # with the first 00:12 is meaningless
    # Thus 00:16 equivilent to 4 seconds
    actdata = actdata[int(4*fps):,:]
    # Add the last frame activation to actdata
    actdata_lastframe = np.tile(actdata[-1,:][None,:],(int(4*fps),1))
    actdata = np.concatenate((actdata,actdata_lastframe),axis=0)
    timept = actdata.shape[0]
    actlength = (timept)/(fps)
    frametimes = np.arange(0,actlength,tr)
    onset = np.linspace(0,(timept)/fps,timept)
    duration = np.array([1/fps]*timept)
    cnn_hrf_signal = analyzer.convolve_hrf(actdata, onset, duration, int(actlength/tr), tr, ops=ops)
    cnn_hrf_signal = stats.zscore(cnn_hrf_signal,axis=0)
    return cnn_hrf_signal


def _detrend(ciftidata, order=4):
    """
    """
    outdata = np.zeros_like(ciftidata)
    for i in range(ciftidata.shape[1]):
        outdata[:,i] = detrend(ciftidata[:,i], order=order)
    return outdata


def prepare_fMRI(subject='subject1', vidseg=['seg1']):
    """
    Read data
    Detrend
    
    mridata: remove the first volume and the last 4 volumes
    """
    print('Prepare actual fMRI')
    mri_range = (1,241)
    parpath = '/nfs/e3/natural_vision/liuzm_movie/source_data/video_fmri_dataset/'
    for i, vs in enumerate(vidseg):
        print('...fMRI data for video segment {}'.format(vs))
        mridata1, header = cifti.read(pjoin(parpath, subject, 'fmri', vs, 'cifti', vs+'_1_Atlas.dtseries.nii'))
        mridata1 = mridata1[mri_range[0]:mri_range[1],:59412]
        mridata2, header = cifti.read(pjoin(parpath, subject, 'fmri', vs, 'cifti', vs+'_2_Atlas.dtseries.nii'))
        mridata2 = mridata2[mri_range[0]:mri_range[1],:59412]
        mridata1_detrend = _detrend(mridata1)
        mridata1_detrend = stats.zscore(mridata1_detrend,axis=0)
        mridata2_detrend = _detrend(mridata2)
        mridata2_detrend = stats.zscore(mridata2_detrend,axis=0)
        if i == 0:
            mridata = (mridata1_detrend+mridata2_detrend)/2
        else:
            mridata = np.append(mridata, (mridata1_detrend+mridata2_detrend)/2, axis=0)
    return mridata, header

def project_fmri(modelpara_path, fmri_subj, vidseg, outunit=1000):
    """
    """
    outfmri = np.zeros((outunit, 59412))
    alexnet = prepare_cnn(modelpara_path)
    cnn_hrf_signal = []
    for i, vseg in enumerate(vidseg):
        actval, fps = extract_video_frames(alexnet, vseg)
        cnn_hrf_tmp = get_cnn_hrf_signal(actval, fps=fps)
        cnn_hrf_signal.extend(cnn_hrf_tmp)
    cnn_hrf_signal = np.array(cnn_hrf_signal)
    for sj in fmri_subj:   
        print('run subject {}'.format(sj)) 
        fmridata, _ = prepare_fMRI(sj, vidseg) 
        outfmri_tmp, _ = tools.pearsonr(cnn_hrf_signal.T, fmridata.T)
        outfmri+=outfmri_tmp
    outfmri/=len(fmri_subj)
    return outfmri
