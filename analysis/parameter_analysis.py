import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class ParamSet(object):
    """"""
    def __init__(self,paramNtrial,labelNtrail):
        assert paramNtrial.shape[0] == len(labelNtrail)

        self.paramset = paramNtrial
        self.labelset = labelNtrail
        self.pvalue = None
        self.pSignIndex = None
        self.effectSize = None
        self.contributionRate = None
        self.p_maxsignParam = None

    def param_analysis(self):
        self.param_ttest()
        self.param_effect_size()
        self.cal_contribution()
        self.sort_contribution()

    def param_ttest(self,method='Bonferrni'):
        label_0 = np.argwhere(self.labelset == 0).astype('int32')
        label_1 = np.argwhere(self.labelset == 1).astype('int32')

        t_sum = []
        p_sum = []
        for i in range(4092):
            x_5000 = self.paramset[:, i]
            x_0 = x_5000[label_0]
            x_1 = x_5000[label_1]

            t, p = stats.ttest_ind(x_0, x_1)
            t_sum.append(t)
            p_sum.append(p)
        self.pvalue = p_sum
        if method == 'Bonferrni':
            #self.pSignIndex = np.squeeze(np.argwhere(np.array(p_sum) < (0.05/4092)))
            self.pSignIndex = []
            temp = []
            for i, p in enumerate(p_sum):
                if p < (0.05/4092):
                    self.pSignIndex.append(i)
                    temp.append(self.pvalue[i])
            self.p_minsignParam = max(temp)
        # elif method == 'fdr':  # the part has bug.
        #     self.pSignIndex = sm.stats.multipletests(p_sum,alpha=0.05,method='fdr_bh')[0]
        #     self.p_minsignIndex = np.squeeze(np.argwhere(self.pSignIndex == self.pSignIndex.max()))
        else:
            print('The correct method are not supported!')
        return self.pvalue, self.pSignIndex, t_sum

    def param_effect_size(self,abs=False):

        d = lambda x1, x2: (x1.mean() - x2.mean()) / np.sqrt(((np.std(x1))**2 + (np.std(x2))**2)/2)
        label_0 = np.argwhere(self.labelset == 0).astype('int32')
        label_1 = np.argwhere(self.labelset == 1).astype('int32')

        self.effectSize = []
        for i in range(4092):
            x_5000 = self.paramset[:, i]
            x_0 = x_5000[label_0]
            x_1 = x_5000[label_1]

            if abs:
                dis = np.abs(d(x_0,x_1))
            else:
                dis = d(x_0, x_1)
            self.effectSize.append(dis)
        self.effectSize = np.array(self.effectSize)
        return self.effectSize

    def cal_contribution(self):
        #return contritbuion, contributionRate
        # calculate the sum of contribution and calculate the contribution percentage of signal parameter.
        sumContrib = self.effectSize.sum()
        self.contributionRate = self.effectSize / sumContrib
        return self.contributionRate

    def sort_contribution(self):
        # sort the contribution and calculate the cumulative contribution along the decreasing with contribution.
        self.indexSorted = np.argsort(-self.contributionRate)
        paramContribSorted = self.contributionRate[self.indexSorted]
        self.cumSumContriRate = np.cumsum(paramContribSorted)
        return self.indexSorted, self.cumSumContriRate

    def get_high_contritbuion_param(self, threshold):
        num_paramOverthreshold = len(np.argwhere(self.cumSumContriRate < threshold))+1
        IndexOverthreshold = self.indexSorted[:num_paramOverthreshold]
        return IndexOverthreshold


def data2diffscale(data,index=None):
    dataSubScale = []
    if type(index) is np.ndarray:
        dataSubScale.append(data[index[index<12]])
        dataSubScale.append(data[index[(index>=12) & (index<60)]])
        dataSubScale.append(data[index[(index>=60) & (index<252)]])
        dataSubScale.append(data[index[(index>=252) & (index<1020)]])
        dataSubScale.append(data[index[(index>=1020) & (index<4092)]])
    else:
        dataSubScale.append(data[:12])
        dataSubScale.append(data[12:60])
        dataSubScale.append(data[60:252])
        dataSubScale.append(data[252:1020])
        dataSubScale.append(data[1020:])

    return dataSubScale

def index2diffscale(index):
    indexSubScale = []
    indexSubScale.append(index[index<12])
    indexSubScale.append(index[(index>=12) & (index<60)])
    indexSubScale.append(index[(index>=60) & (index<252)])
    indexSubScale.append(index[(index>=252) & (index<1020)])
    indexSubScale.append(index[(index>=1020) & (index<4092)])
    return indexSubScale


if __name__ == '__main__':
    param_20000 = np.load(r'D:\cnnface\Data_sorted\vggface\raw/params_20000.npy')
    label_20000 = np.load(r'D:\cnnface\Data_sorted\vggface\raw/vgg_activation_label_20000.npy')
    vggParamSet = ParamSet(param_20000,label_20000)
    vggParamSet.param_analysis()

    np.save(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/pvalue.npy',vggParamSet.pvalue)
    np.save(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/d.npy',vggParamSet.effectSize)
    np.save(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/pSignIndex.npy',vggParamSet.pSignIndex)
    np.save(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/contribution.npy',vggParamSet.contributionRate)
    np.save(r'D:\cnnface\Data_sorted\vggface\param_analysis\data/p_maxsignIndex.npy',np.array([798]))

