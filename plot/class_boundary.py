import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('darkgrid')


def class_boundary(label_prob):
    class0 = label_prob[:, 0]
    class1 = label_prob[:, 1]
    x = np.arange(1, len(class0)+1)

    sns.lineplot(x, class0)
    sns.lineplot(x, class1)
    plt.legend(['female', 'male'])
    plt.title('')
    plt.show()

    baseline = 0.5
    distance = np.abs(class0 - baseline)
    print('The Picture with a classification probability closest to 50%:',np.argmin(distance)+1)

    return distance


