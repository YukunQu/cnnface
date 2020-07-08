import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def generateSingleSinusoid(img_size, angle, phase, amplitude):
    angle = np.radians(angle)
    sinepath = (np.linspace(0, 2, img_size))[:,np.newaxis].repeat(img_size,axis=1)
    sinusoid = (sinepath*np.sin(angle) + sinepath.T*np.cos(angle)) * 2 * np.pi
    sinusoid = amplitude * np.sin(sinusoid + phase)

    return sinusoid


def generatePatches(img_size,nscale=5):
    scales = [2**i for i in range(1,nscale+1)]
    NumPatch = [(scale/2)**2 for scale in scales]
    patches = {}

    for scale,numpatch in zip(scales,NumPatch):
        length = int(scale/2)
        patch_size = img_size / length
        angle = [0, 30, 60, 90, 120, 150]
        phases = [0, np.pi / 2]
        sinusoid_combined = []

        for p in phases:
            for a in angle:
                sinusoid = generateSingleSinusoid(patch_size, a, p)
                sinusoid_combined.append(sinusoid)
        patch = np.array(sinusoid_combined)
        patch = np.tile(patch, (1, length, length))
        patches[scale] = patch
    return patches


def generateNoise(img_size,patches, nscale=5):

    scales = [2**i for i in range(1,nscale+1)]
    NumPatch = [(scale / 2) ** 2 for scale in scales]
    NumParam = [int(num*12) for num in NumPatch]
    params = {}

    for scale, numpatch, numpara in zip(scales,NumPatch,NumParam):
        length = int(scale/2)
        patch_size = img_size / length
        param = np.random.uniform(-1, 1, size=numpara)
        param = param.reshape((12, length, length))
        param = param.repeat(patch_size, axis=1).repeat(patch_size,axis=2)
        params[scale] = param

    noise = [patches[scale] * params[scale] for scale in scales]
    noise = np.array(noise)
    noise = np.sum(np.sum(noise, axis=0), axis=0)
    return noise, params


if __name__=='__main__':
    param_n = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
    para = param_n[0][:12]
    patch_size = 512
    angle = [0, 30, 60, 90, 120, 150]
    phases = [0, np.pi / 2]
    sinusoid_combined = []

    i = 0
    for a in angle:
        for p in phases:
            sinusoid = generateSingleSinusoid(patch_size, a, p, para[i])
            sinusoid_combined.append(sinusoid)
            i += 1
    patch = np.array(sinusoid_combined)
    patch = np.mean(patch,axis=0)
    # for p in patch:
    #     plt.imshow(p,'gray')
    #     plt.axis('off')
    #     plt.show()
    plt.imshow(patch,'gray')
    plt.axis('off')
    plt.show()