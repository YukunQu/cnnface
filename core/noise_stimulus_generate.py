import numpy as np


def generateSingleSinusoid(img_size, angle, phase):
    angle = np.radians(angle)
    sinepath = (np.linspace(0,2,img_size))[:,np.newaxis].repeat(img_size,axis=1)
    sinusoid = (sinepath*np.sin(angle) + sinepath.T*np.cos(angle)) * 2 *np.pi
    sinusoid = np.sin(sinusoid + phase)

    return sinusoid


def generatePatch(img_size):
    angle = [0,30,60,90,120,150]
    phases = [0,np.pi/2]
    sinusoid_combined = []

    for p in phases:
        for a in angle:
            sinusoid = generateSingleSinusoid(img_size,a,p)
            sinusoid_combined.append(sinusoid)
    sinusoid_combined = np.array(sinusoid_combined)    #actually need to multiply factors
    #sinusoid_combined = np.sum(sinusoid_combined,axis=0)# for test, no need to sum()
    return sinusoid_combined


def generatePatches(img_size,nscale=5):
    scales = [2**i for i in range(1,nscale+1)]
    NumPatch = [(scale/2)**2 for scale in scales]
    patches = {}

    for scale,numpatch in zip(scales,NumPatch):
        img_size = img_size / (scale/2)
        patch = generatePatch(img_size)
        patch = np.tile(patch,(1,(scale/2),(scale/2)))
        patches[scale] = patch

    return patches


def generateNoise(img_size,patches,nscale=5):

    scales = [2**i for i in range(1,nscale+1)]
    NumPatch = [(scale / 2) ** 2 for scale in scales]
    NumParam = [ num*12 for num in NumPatch]
    params = {}

    for scale,numpatch,numpara in zip(scales,NumPatch,NumParam):

        img_size = img_size / (scale/2)
        patch = generatePatch(img_size)
        patch = np.tile(patch,(1,(scale/2),(scale/2)))
        patches[scale] = patch

        param = np.random.randn(numpara)
        param = param.reshape(12,numpatch)
        param = param.repeat(img_size,axis=0).repeat(img_size,axis=1)
        params[scale] = param

    noise = [patches[scale] * params[scale] for scale in scales]
    noise = np.sum(np.array(noise),axis=0)

    return noise,params