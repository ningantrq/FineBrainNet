import os
import csv
import numpy as np
from sklearn import metrics


class LoggerBrainNetFormer(object):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)

    def __call__(self, **kwargs):
        if len(kwargs)==0:
            self.get()
        else:
            self.add(**kwargs)

    def _initialize_metric_dict(self):
        return {'pred':[], 'true':[]}

    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:
                self.samples = {}
                #@ samples是一个k*n的dict 存储了中间的信息
                for _k in range(self.k_fold):
                    self.samples[_k] = self._initialize_metric_dict()
            else:
                self.samples[k] = self._initialize_metric_dict()

    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for sample, value in kwargs.items():
                self.samples[sample].append(value)
        else:
            assert k in list(range(self.k_fold))
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)

    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            true = np.concatenate(self.samples['true'])
            pred = np.concatenate(self.samples['pred'])
        else:
            if k is None:
                true, pred, prob = {}, {}, {}
                for k in range(self.k_fold):
                    true[k] = np.concatenate(self.samples[k]['true'])
                    pred[k] = np.concatenate(self.samples[k]['pred'])
            else:
                true = np.concatenate(self.samples[k]['true'])
                pred = np.concatenate(self.samples[k]['pred'])

        if initialize:
            self.initialize(k)

        return dict(true=true, pred=pred)


    def evaluate(self, k=None, initialize=False, option='mean'):
        samples = self.get(k)
        if not self.k_fold is None and k is None:
            if option=='mean': aggregate = np.mean
            elif option=='std': aggregate = np.std
            else: raise
            accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
            precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
            recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
        else:
            accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
            precision = metrics.precision_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
            recall = metrics.recall_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
            # roc_auc = 0.5
            # ! 这里有问题 炸炉了(已排查 是数据的问题)
        if initialize:
            self.initialize(k)

        return dict(accuracy=accuracy, precision=precision, recall=recall)


    def to_csv(self, targetdir, k=None, initialize=False):
        metric_dict = self.evaluate(k, initialize)
        append = os.path.isfile(os.path.join(targetdir, 'metric.csv'))
        with open(os.path.join(targetdir, 'metric.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(['fold'] + [str(key) for key in metric_dict.keys()])
            writer.writerow([str(k)]+[str(value) for value in metric_dict.values()])
            if k is None:
                writer.writerow([str(k)]+list(self.evaluate(k, initialize, 'std').values()))
