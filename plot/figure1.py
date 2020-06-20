import numpy as np
import matplotlib.pyplot as plt
from cnnface.stimuli.image_manipulate import img_similarity,nor


def ci_show(ci,savepath=False,colorbar=True):
    ci = nor(ci)
    plt.clf()
    plt.imshow(ci,cmap='jet')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if colorbar is True:
        plt.colorbar()
    if savepath is False:
        plt.show()
    else:
     plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)


if __name__ == '__main__':
    ci_human = np.load(r'D:\cnnface\Data_sorted\human\ci\data/cis_human.npy')
    #ci_vgg_act = np.load(r'D:\cnnface\Data_sorted\vggface_act\ci\data/ci_vgg_act.npy')
    ci_vgg = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/ci_vgg.npy')
    similarity = img_similarity(ci_human, ci_vgg, 'pearsonr')
    # ci_alexnet = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\ci_result/ci_alexnet.npy')
    # similarity = img_similarity(ci_human,ci_alexnet,'pearsonr')

    print('The similarity of two CIs:', similarity[0])
    print('p value:', np.round(similarity[1]))

    # ci_show(ci_vgg_act,r'F:\研究生资料库\项目五：AI\文章图\img\sp2/ci_vgg_act.jpg')
    # ci_show(ci_vgg,r'F:\研究生资料库\项目五：AI\文章图\img\sp2/ci_vgg.jpg')
   # np.save(r'F:\研究生资料库\项目五：AI\文章图\img\sp2/similarity.npy',similarity)
    #ci_show(ci_vgg,r'F:\研究生资料库\项目五：AI\文章图\img\Figure1/ci_vgg.jpg')