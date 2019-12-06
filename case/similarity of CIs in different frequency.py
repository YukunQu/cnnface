# Calculate the CIs Correlation between human and CNN in different frequency scales

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cnnface.analysis.generate_ci import generateCI,recon_face
from cnnface.stimulus.Image_process import nor,img_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# generate the human cis in different frequency scales
paramci_human = np.load(r'D:\cnnface\gender_analysis\human_result\CIs/param_ci_human.npy')
cis_human = generateCI(paramci_human, [2, 4, 8, 16, 32])

# generate the cnn cis in different frequency scales
paramci_cnn = np.load(r'D:\cnnface\gender_analysis\CI_analysis/param_ci_cnn.npy')
cis_cnn = generateCI(paramci_cnn, [2, 4, 8, 16, 32])

scales = [2, 4, 8, 16, 32]
face_add_human = []
face_sub_human = []
# generate the reconstruct face in different frequency scales
baseface = Image.open(r'D:\cnnface\gender_analysis\face_template\gray/baseface.jpg')
for scale,img in zip(scales,cis_human):
    picSavePath_add = r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face\human/face_{}_add.jpg'.format(scale)
    picSavePath_sub = r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face\human/face_{}_sub.jpg'.format(scale)
    scaleIndex = 45/img.max()
    img_add, img_sub = recon_face(baseface, img, scale=scaleIndex)
    face_add_human.append(np.array(img_add))
    face_sub_human.append(np.array(img_sub))
    # img_add.save(picSavePath_add,quality=95)
    # img_sub.save(picSavePath_sub,quality=95)

face_add_cnn = []
face_sub_cnn = []
for scale,img in zip(scales,cis_cnn):
    picSavePath_add = r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face\cnn/face_{}_add.jpg'.format(scale)
    picSavePath_sub = r'D:\cnnface\gender_analysis\Result\ci_correlation\recon_face\cnn/face_{}_sub.jpg'.format(scale)
    scaleIndex = 45 / img.max()
    img_add, img_sub = recon_face(baseface, img, scale=scaleIndex)
    face_add_cnn.append(np.array(img_add))
    face_sub_cnn.append(np.array(img_sub))
    # img_add.save(picSavePath_add,quality=95)
    # img_sub.save(picSavePath_sub,quality=95)

# generate the ci in diifferent frequency scales
for scale,img in zip(scales,cis_human):
    savePath = r'D:\cnnface\gender_analysis\Result\ci_correlation' \
               r'\CIs Correlation between human and CNN in different frequency scales\cnn/ci_cnn_{}.jpg'.format(scale)
    plt.imshow(img,cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(savePath, cmap='jet')

for scale,img in zip(scales,cis_cnn):
    savePath = r'D:\cnnface\gender_analysis\Result\ci_correlation' \
               r'\CIs Correlation between human and CNN in different frequency scales\human/ci_human_{}.jpg'.format(scale)
    plt.imshow(img,cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(savePath,  cmap='jet')


cis_human = [nor(ci) for ci in cis_human]
cis_cnn = [nor(ci) for ci in cis_cnn]
cisCorrelationBetweenCnn2Human = [img_similarity(ci_cnn,ci_human,'pearsonr',r_p=True)
                                  for ci_cnn,ci_human in zip(cis_cnn,cis_human)]
np.save(r'D:\cnnface\gender_analysis\Result\ci_correlation/cisCorrelationBetweenCnn2HumanDiffScale.npy',cisCorrelationBetweenCnn2Human)

sns.set_style('darkgrid')
cisCorr = np.array(cisCorrelationBetweenCnn2Human)
sns.lineplot(scales,cisCorr[:, 0])
plt.tick_params(labelsize=15)
plt.xlabel('frequency scale',fontsize=15)
plt.ylabel('Correlation',fontsize=15)
plt.savefig(r'D:\cnnface\gender_analysis\Result\ci_correlation/correlationDiffScale.jpg')
plt.show()

faceCorradd = [img_similarity(face_cnn,face_human,'pearsonr',r_p=True) for face_cnn,face_human in zip(face_add_cnn,face_add_human)]
faceCorrsub = [img_similarity(face_cnn,face_human,'pearsonr',r_p=True) for face_cnn,face_human in zip(face_sub_cnn,face_sub_human)]
