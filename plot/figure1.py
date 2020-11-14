import numpy as np
import matplotlib.pyplot as plt
from cnnface.stimuli.image_manipulate import img_similarity, nor


#%%
def ci_show(ci, savepath=False, colorbar=True):
    ci = nor(ci)
    plt.clf()
    plt.imshow(ci,cmap='jet')
    plt.axis('off')
    cbar = plt.colorbar()
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_size(24)
    cbar.set_ticks(np.arange(0, 1.1, 0.5))
    cbar.set_ticklabels(['0', '0.5', '1'])

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # if colorbar is True:
    #     plt.colorbar()
    if savepath is False:
        plt.show()
    else:
     plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    ci_human = np.load(r'D:\cnnface\Data_sorted\human\ci\data/ci_human.npy')
    ci_vgg = np.load(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\ci\alexnet_ci_act.npy')


    similarity = img_similarity(ci_human, ci_vgg, 'pearsonr')

    print('The similarity of two CIs:', np.round(similarity[0], 2))
    print('p value:', np.round(similarity[1], 2))

    np.savetxt(r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\result1\similarity/similarity_alexnet_act.txt', similarity)
    ci_show(ci_vgg, r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\result1/ci_alexnet_act.jpg')
    ci_show(ci_human, r'D:\cnnface\analysis_for_reply_review\analysis\new dataset\result\result1/ci_human.jpg')