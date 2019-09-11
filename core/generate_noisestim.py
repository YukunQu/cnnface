import numpy as np
from PIL import Image
import time

mean = [124.82445313, 99.1312664, 87.52420379]
std  = [71.39276188, 64.86421609, 62.8511147]

time0 =time.time()
for i in range(100000):
    noise_img_r = np.uint8(np.random.normal(mean[0],std[0],(224,224,1)))
    noise_img_g = np.uint8(np.random.normal(mean[1],mean[1],(224,224,1)))
    noise_img_b = np.uint8(np.random.normal(mean[2],mean[2],(224,224,1)))

    noise_img = np.concatenate((noise_img_r,noise_img_g),axis=2)
    noise_img = np.concatenate((noise_img,noise_img_b),axis=2)

    noise_img = Image.fromarray(noise_img)
    noise_img.save('D:/cnnface/noise_picture/noise_img_{}.jpg'.format(i))
print(time.time()-time0)