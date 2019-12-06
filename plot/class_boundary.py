import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def class_boundary(label_prob):
    class0 = label_prob[:, 0]
    class1 = label_prob[:, 1]
    x = np.arange(1, len(class0)+1)

    sns.lineplot(x, class0)
    sns.lineplot(x, class1)
    plt.show()
