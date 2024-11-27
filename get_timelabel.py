#%%
import os
import scipy.io as scio
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
import torch
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold
from dataset import DatasetHCPTask
import torch.nn as nn
from einops import rearrange, repeat
# %%
#@ 每个timeseries的最大长度
Tmax = {'WM':405,'SOCIAL':274,'RELATIONAL':232,'MOTOR':284,'LANGUAGE':316,'GAMBLING':253,'EMOTION':176}
#@ subtask block name
file_list = {'WM':['0bk_body.txt','0bk_faces.txt','0bk_places.txt','0bk_tools.txt','2bk_body.txt','2bk_faces.txt','2bk_places.txt','2bk_tools.txt'],\
    'GAMBLING':['win.txt','loss.txt'],\
        'MOTOR':['cue.txt','lf.txt','rf.txt','lh.txt','rh.txt','t.txt'],\
            'LANGUAGE':['story.txt','math.txt'],\
                'SOCIAL':['mental.txt','rnd.txt'],\
                    'RELATIONAL':['relation.txt','match.txt'],\
                        'EMOTION':['fear.txt','neut.txt']}
#@ 对每个任务的label进行编码
l = 1
subtask_map = {} #@ 建立映射表
for v in file_list.values():
    for subtask in v:
        subtask_map[subtask[:-4]] = l
        l += 1


#@ 生成每个时刻的标签dict（目前为空，表示全为rest）
subtask_labels = {}
for task, maxt in Tmax.items():
    subtask_labels[task] = np.array([0]*maxt)
    print(task, maxt)

#@ 选取的evs文件目录
datadir = ['/home/zhangke/dataset/HCP/fMRI/task fmri/WM/Preprocess_Data/966975/MNINonLinear/Results/tfMRI_WM_LR/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/GAMBLING/Preprocess_Data/100307/MNINonLinear/Results/tfMRI_GAMBLING_LR/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/MOTOR/Preprocess_Data/994273/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs',\
# '/home/zhangke/dataset/HCP/fMRI/task fmri/LANGUAGE/Preprocess_Data/995174/MNINonLinear/Results/tfMRI_LANGUAGE_RL/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/LANGUAGE/Preprocess_Data/995174/MNINonLinear/Results/tfMRI_LANGUAGE_LR/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/SOCIAL/Preprocess_Data/100610/MNINonLinear/Results/tfMRI_SOCIAL_LR/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/RELATIONAL/Preprocess_Data/996782/MNINonLinear/Results/tfMRI_RELATIONAL_LR/EVs',\
'/home/zhangke/dataset/HCP/fMRI/task fmri/EMOTION/Preprocess_Data/100206/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs']

#%%
#@ 生成详细subtask还是仅仅rest和task区分
task_detail = True
need_saving = True

if(task_detail):
    #@ 生成详细subtask标签
    for datadir_ in datadir:
        print(datadir_)
        task_name = datadir_.split('/')[7]

        #@ 创建空白表单
        subtask_table = pd.DataFrame(columns=[0, 1, 2, 'subtask']) 
        for file_ in file_list[task_name]:
            # print(file_)
            df =  pd.read_table(os.path.join(datadir_, file_), header=None)
            df['subtask'] = file_[:-4]
            subtask_table = pd.concat((subtask_table, df), axis=0)

        subtask_table.rename(columns = {0:'onset',1:'duration',2:'tensity'}, inplace=True)
        subtask_table.sort_values('onset', inplace= True)
        subtask_table.reset_index(inplace=True)
        subtask_table[['onset','duration']] = subtask_table[['onset','duration']]  / 0.72 #@ 每720ms进行一次采样
        subtask_table.onset = subtask_table.onset.apply(int)
        subtask_table.duration = subtask_table.duration.apply(int)
        subtask_table['end'] = subtask_table.onset + subtask_table.duration
        subtask_table['next_start'] = subtask_table.onset.shift(-1)
        subtask_table.next_start.iloc[-1] = Tmax[task_name]
        subtask_table['period'] = subtask_table.next_start - subtask_table.end
        print(task_name)
        print(subtask_table)
        for index,row in subtask_table.iterrows():
            for i in range(row.onset, min(row.end, Tmax[task_name])):
                subtask_labels[task_name][i] = subtask_map[row.subtask]
            print(index, row.onset, row.end)
    if(need_saving):
        np.save('data/subtask_labels_detail.npy', subtask_labels)
else:
    #@ 生成task rest标签
    for datadir_ in datadir:
        print(datadir_)
        task_name = datadir_.split('/')[7]

        #@ 创建空白表单
        subtask_table = pd.DataFrame(columns=[0, 1, 2, 'subtask']) 
        for file_ in file_list[task_name]:
            # print(file_)
            df =  pd.read_table(os.path.join(datadir_, file_), header=None)
            df['subtask'] = file_[:-4]
            subtask_table = pd.concat((subtask_table, df), axis=0)

        subtask_table.rename(columns = {0:'onset',1:'duration',2:'tensity'}, inplace=True)
        subtask_table.sort_values('onset', inplace= True)
        subtask_table.reset_index(inplace=True)
        subtask_table[['onset','duration']] = subtask_table[['onset','duration']]  / 0.72
        subtask_table.onset = subtask_table.onset.apply(int)
        subtask_table.duration = subtask_table.duration.apply(int)
        subtask_table['end'] = subtask_table.onset + subtask_table.duration
        subtask_table['next_start'] = subtask_table.onset.shift(-1)
        subtask_table.next_start.iloc[-1] = Tmax[task_name]
        subtask_table['period'] = subtask_table.next_start - subtask_table.end
        print(task_name)
        print(subtask_table)
        for index,row in subtask_table.iterrows():
            for i in range(row.onset, min(row.end, Tmax[task_name])):
                subtask_labels[task_name][i] = 1
            print(index, row.onset, row.end)
    if(need_saving):
        np.save('data/subtask_labels_2.npy', subtask_labels)
# %%
# natt = np.load('/home/zhangke/slh/BrainNetFormer/result/task_origin/attention/0/node_attention/EMOTION.npy')
