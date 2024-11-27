import os

from sklearn import svm
from torch.nn import CrossEntropyLoss

import util
import random
import torch
import numpy as np
# from model import *
from BrainNetFormer import *
from dataset import *
from tqdm import tqdm
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from sklearn.neural_network import MLPClassifier

def base_test(argv):

    # define dataset
    if argv.dataset == 'rest':
        dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    elif argv.dataset == 'task':
        dataset = DatasetHCPTask(argv.sourcedir, subtask_type=argv.subtask_type, roi=argv.roi,
                                 dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    else:
        raise

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)
    subtask_nums = 2 if argv.subtask_type == 'simple' else 25
    print('subtask_nums: ', subtask_nums)
    logger_subtask = util.base_logger.LoggerBrainNetFormer(argv.k_fold, subtask_nums)

    dataset.train_final = True
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # @ ************************************ start experiment **************************************** @#
    for k in range(0, argv.k_fold):
        # make directories per fold
        # set dataloader
        dataset.set_fold(k, train=True)
        #model = svm.SVC(C=10, kernel='sigmoid')
        model = MLPClassifier(hidden_layer_sizes=(64,1))
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer and learning rate scheduler
        ''' 训练阶段 '''
        # @ ******************************** start training ***********************************@#
        for epoch in range(0, argv.num_epochs):
            # @ ********************************* train ***************************************@#
            dataset.set_fold(k, train=True)
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):  #
                # process input data
                # @ 预处理数据
                # t = x['timeseries'].permute(1, 0, 2)
                t=x['timeseries']
                t=t[0]
                label = x['label']
                # subtask_label = x['subtask_label'].permute(1,0)[sampling_points].permute(1,0)
                subtask_label = x['subtask_label']
                subtask_label = rearrange(subtask_label, 'b n -> (b n)')
                subtask_label=subtask_label.T

                model = model.fit(t,subtask_label)

        subtask_nums_list = argv.subtask_nums
        subtask_nums_list = list(subtask_nums_list.values())

        logger_subtask.initialize(k)
        dataset.set_fold(k, train=False)

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data

                t = x['timeseries']
                t=t[0]

                subtask_label = x['subtask_label']
                subtask_label = rearrange(subtask_label, 'b n -> (b n)')
                subtask_label=subtask_label.T
                pred=model.predict(t)

                #@ subtask
                logger_subtask.add(k=k, pred=pred,
                                   true=subtask_label)


        samples_subtask = logger_subtask.get(k)
        metrics_subtask = logger_subtask.evaluate(k)

        print("subtask:", metrics_subtask)

        logger_subtask.to_csv(argv.targetdir, k)

    logger_subtask.to_csv(argv.targetdir)
    final_metrics = logger_subtask.evaluate()
    print('subtask:', final_metrics)





