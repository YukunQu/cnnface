import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_dev(train_acc_dev,val_acc_dev):
    epoch = range(len(train_acc_dev))
    sns.lineplot(epoch, train_acc_dev)
    sns.lineplot(epoch, val_acc_dev)
    plt.legend(['train', 'val'])
    plt.show()