import numpy as np


def generateSingleSinusoid(img_size, cycles, angle, phase):
    angle = np.radians(angle)
    sinepath = (np.linspace(0,cycles,img_size))[:,np.newaxis].repeat(img_size,axis=1)
    sinusoid = (sinepath*np.sin(angle) + sinepath.T*np.cos(angle)) * 2 *np.pi
    sinusoid = np.sin(sinusoid + phase)
    return sinusoid


def generateCombinSinusoid(img_size,cycles,angle,phase):
    sinusoid_combined = []
    for p in phase:
        for a in angle:
            sinusoid = generateSingleSinusoid(img_size,cycles,a,p)
            sinusoid = np.random.randn() * sinusoid
            sinusoid_combined.append(sinusoid)
    sinusoid_combined = np.array(sinusoid_combined)
    sinusoid_combined = np.sum(sinusoid_combined,axis=0)
    sinusoid_combined = (sinusoid_combined-sinusoid_combined.min())/(sinusoid_combined.max()-sinusoid_combined.min())
    sinusoid_combined = sinusoid_combined *255
    return sinusoid_combined
