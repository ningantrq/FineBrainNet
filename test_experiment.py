import os

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

# subtask_nums=argv.subtask_nums[dataset.task_list[index]]
def step(model, criterion, dyn_t, dyn_a, sampling_points, sampling_endpoints, t, a, label
         , train=True, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    # @ main task prediction
    logit, attention = model(dyn_t.to(device), dyn_a.to(device), t.to(device), a.to(device),
                                            sampling_endpoints)
    loss = criterion(logit, label.to(device))

    # # @ subtask prediction
    # loss_subtask = criterion(logit_subtask, subtask_label.to(device))
    # loss += loss_subtask * reg_subtask

    # @ optimize model
    if (train):
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    return logit, loss, attention

def sub_step(model, criterion, sampling_points, sampling_endpoints, t, a, label,subtask_label
         , train=True, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    subtask_label_reverse_dict= {'EMOTION': 1, 'GAMBLING': 3, 'LANGUAGE': 5, 'MOTOR': 7, 'RELATIONAL': 13,
                                 'SOCIAL': 15, 'WM': 17}
    if optimizer is None:
        model.eval()
    else:
        model.train()

    # @ main task prediction
    # logit, attention = model(t.to(device))
    # loss = criterion(logit, label.to(device))

    # # @ subtask prediction

    #子任务结果向量维度拓展
    sub_batchsize=subtask_label.size(0)
    logit_subtask=torch.zeros(sub_batchsize,25)

    logit_subtask_ori=model(t.to(device),label)
    for batch in range(sub_batchsize):
        logit_subtask[batch,0]=logit_subtask_ori[batch,0]
        insert_begin=subtask_label_reverse_dict[label]
        for index in range(1,logit_subtask_ori.size(1)):
            logit_subtask[batch,insert_begin+index-1]=logit_subtask_ori[batch,index]

    #子任务label维度拓展
    subtask_label_final=subtask_label
    for index in range(sub_batchsize):
        if(subtask_label_final[index]!=0):
            insert_begin = subtask_label_reverse_dict[label]
            subtask_label_final[index]=insert_begin+subtask_label[index]-1

    loss = criterion(logit_subtask, subtask_label_final.to(device))


    # @ optimize model
    if (train):
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    return logit_subtask, loss

def Test(argv):
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    if argv.dataset == 'rest':
        dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    elif argv.dataset == 'task':
        dataset = DatasetHCPTask(argv.sourcedir, subtask_type=argv.subtask_type, roi=argv.roi,
                                 dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    else:
        raise
    dataset.train_final = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    num_timepoints = len(list(range(0, argv.dynamic_length - argv.window_size, argv.window_stride)))
    subtask_nums = 2 if argv.subtask_type == 'simple' else 25

    logger = util.logger.LoggerBrainNetFormer(argv.k_fold, dataset.num_classes)
    logger_subtask = util.logger.LoggerBrainNetFormer(argv.k_fold, subtask_nums)

    for k in range(argv.k_fold):
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        subtask_nums_list = argv.subtask_nums
        subtask_nums_list = list(subtask_nums_list.values())

        model = MaintaskPredictor(
            n_region=dataset.num_nodes,  # n_region是顶点的个数N
            hidden_dim=argv.hidden_dim,  # hidden_dim是特征的维度D
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            window_size=argv.window_size)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'Mainmodel', str(k), 'model.pth')))

        sub_model = SubtaskPredictor(
            n_region=dataset.num_nodes,  # n_region是顶点的个数N
            hidden_dim=argv.hidden_dim,  # hidden_dim是特征的维度D
            subtask_nums_dict=argv.subtask_nums)
        sub_model.to(device)
        sub_model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'Submodel', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss()

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
        sub_summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'Subsummary', str(k), 'test'))

        logger.initialize(k)
        logger_subtask.initialize(k)

        dataset.set_fold(k, train=False)

        loss_accumulate = 0.0
        loss_subtask_accumulate = 0.0

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data
                dyn_t, sampling_points = util.bold.process_dynamic_t(x['timeseries'], argv.window_size,
                                                                     argv.window_stride)
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                      argv.window_stride)
                sampling_endpoints = [p + argv.window_size for p in sampling_points]
                # print(dyn_t.shape, dyn_a.shape)
                a = util.bold.process_static_fc(x['timeseries'])
                t = x['timeseries'].permute(1, 0, 2)
                # a =
                label = x['label']
                subtask_label = x['subtask_label'].permute(1, 0)[sampling_points].permute(1, 0)
                subtask_label = x['subtask_label']
                subtask_label = rearrange(subtask_label, 'b n -> (b n)')
                for index in range(len(subtask_label)):
                    subtask_label[index] = dataset.subtask_label_dict[subtask_label[index].item()]

                # @ 向前传播+反向传播
                # main task
                logit, loss, attention = step(
                    model=model,
                    criterion=criterion,
                    dyn_t=dyn_t,
                    dyn_a=dyn_a,
                    sampling_points=sampling_points,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    a=a,
                    label=label,
                    train=False,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)

                # @ main task result
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(),
                           prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()


                #subtask
                logit_subtask, loss_subtask = sub_step(
                    model=sub_model,
                    criterion=criterion,
                    sampling_points=sampling_points,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    a=a,
                    label=argv.subtask_labels[pred.item()],
                    subtask_label=subtask_label,
                    train=False,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)

                # 子任务label维度拓展
                # subtask_label_reverse_dict = {'EMOTION': 1, 'GAMBLING': 3, 'LANGUAGE': 5, 'MOTOR': 7, 'RELATIONAL': 13,
                #                               'SOCIAL': 15, 'WM': 17}
                # sub_batchsize=subtask_label.size(0)
                # subtask_label_final = subtask_label
                # for index in range(sub_batchsize):
                #     if (subtask_label_final[index] != 0):
                #         insert_begin = subtask_label_reverse_dict[argv.subtask_labels[pred.item()]]
                #         subtask_label_final[index] = insert_begin + subtask_label[index] - 1
                #
                # subtask_label=subtask_label_final

                # @ subtask result
                pred_subtask = logit_subtask.argmax(1)
                prob_subtask = logit_subtask.softmax(1)
                loss_subtask_accumulate += loss_subtask.detach().cpu().numpy()
                logger_subtask.add(k=k, pred=pred_subtask.detach().cpu().numpy(),
                                   true=subtask_label.detach().cpu().numpy(), prob=prob_subtask.detach().cpu().numpy())



        # main task
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss', loss_accumulate / len(dataloader))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']

        summary_writer.flush()
        print(metrics)

        # subtask
        samples_subtask = logger_subtask.get(k)
        metrics_subtask = logger_subtask.evaluate(k)
        sub_summary_writer.add_scalar('loss_subtask', loss_subtask_accumulate / len(dataloader))
        # summary_writer.add_pr_curve('precision-recall_subtask', samples_subtask['true'], samples_subtask['prob'][:, 1])
        sub_summary_writer.add_pr_curve('precision-recall_subtask', samples_subtask['true'],
                                    samples_subtask['prob'])
        [sub_summary_writer.add_scalar(key + '_subtask', value) for key, value in metrics_subtask.items() if
         not key == 'fold']

        sub_summary_writer.flush()
        print("subtask:", metrics_subtask)

        # finalize fold
        logger.to_csv(argv.targetdir, k)
        logger_subtask.to_csv(argv.targetdir, k)

    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    print(final_metrics)

    logger_subtask.to_csv(argv.targetdir)
    final_metrics = logger_subtask.evaluate()
    print('subtask:', final_metrics)
    summary_writer.close()
    sub_summary_writer.close()

    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))
    torch.save(logger_subtask.get(), os.path.join(argv.targetdir, 'Subsamples.pkl'))