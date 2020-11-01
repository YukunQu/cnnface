import numpy as np
import matplotlib.pyplot as plt
from cnnface.stimuli.image_manipulate import img_similarity,nor
#%%

def ci_show(ci,savepath=False,colorbar=True):
    ci = nor(ci)
    plt.clf()
    plt.imshow(ci,cmap='jet')
    plt.axis('off')
    cbar = plt.colorbar()
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_size(24)
    cbar.set_ticks(np.arange(0, 1.1, 0.5))
    cbar.set_ticklabels(['0', '0.5','1'])


    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # if colorbar is True:
    #     plt.colorbar()
    if savepath is False:
        plt.show()
    else:
     plt.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)


if __name__ == '__main__':
    ci_human = np.load(r'D:\cnnface\Data_sorted\human\ci\data/ci_human.npy')
    #ci_vgg_act = np.load(r'D:\cnnface\Data_sorted\vggface_act\ci\data/ci_vgg_act.npy')
    ci_vgg = np.load(r'D:\cnnface\Data_sorted\vggface\ci\data/ci_vgg.npy')
    ci_vgg = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\CI/vggface_ci.npy')

    #ci_vgg16 = np.load(r'D:\cnnface\Data_sorted\vgg16\ci\data/ci.npy')
    similarity = img_similarity(ci_human, ci_vgg, 'pearsonr')
    # ci_alexnet = np.load(r'D:\cnnface\gender_analysis\supplementray_analysis\ci_result/ci_alexnet.npy')
    # similarity = img_similarity(ci_human,ci_alexnet,'pearsonr')

    print('The similarity of two CIs:', similarity[0])
    print('p value:', np.round(similarity[1]))

    ci_show(ci_vgg,r'F:\研究生资料库\项目五：AI\文章图\img\Figure1/ci_vgg.jpg')
    ci_show(ci_human,r'F:\研究生资料库\项目五：AI\文章图\img\Figure1/ci_human.jpg')

