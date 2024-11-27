from matplotlib.pyplot import axis
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


class LayerGIN(nn.Module):
    def __init__(self, n_region, hidden_dim, output_dim, epsilon=True):
        #在强化学习中，Epsilon（ε）通常代表一个探索率（exploration rate）。
        #探索率是强化学习算法中一个关键的超参数，用于平衡探索（exploration）和利用（exploitation）的权衡。
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        # self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU())
    def forward(self, v, a):
        # print(a.shape, v.shape)
        v_aggregate = torch.sparse.mm(a, v)#矩阵乘法（稀疏）
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine

class ModuleTransformer(nn.Module):
    def __init__(self, n_region, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_region, num_heads)
        self.layer_norm1 = nn.LayerNorm(n_region)
        self.layer_norm2 = nn.LayerNorm(n_region)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_region))

    def forward(self, q, k, v):
        x_attend, attn_matrix = self.multihead_attn(q, k, v)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix

class BrainEncoder(nn.Module):
    def __init__(self, n_region, hidden_dim):
        super().__init__()
        #@ For X_enc
        # self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.LayerNorm(hidden_dim),nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),nn.LayerNorm(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=1, dropout=0.1)
        #@ For e_s
        # self.SE = nn.Sequential(nn.Linear(n_region*n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

        self.SE = nn.Sequential(nn.Linear(n_region * n_region, hidden_dim), nn.LayerNorm(hidden_dim),nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),nn.LayerNorm(hidden_dim), nn.ReLU())
    def forward(self, t, a):
        timepoints, batchsize = t.shape[:2]
        X_enc = rearrange(t, 't b c -> (b t) c')#t -> timepoints  b->batchsize  c->?
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=timepoints, b=batchsize)#先合并，MLP后再展开  X_enc B*T*N->B*T*C
        X_enc, (hn, cn) = self.PE(X_enc)#不改变X_enc维度
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc) # t b c

        e_s = self.SE(rearrange(a, 'b n v -> b (n v)')) # b c  a -> batchsize*regions*regions
        return X_enc, e_s

class BrainDecoder(nn.Module):
    def __init__(self, n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size, dropout=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.sparsity = sparsity

        self.cls_token = lambda x: x.sum(0)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()#可以直接多次叠层，ModuleList自带迭代器
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.initial_linear = nn.Linear(window_size, hidden_dim) #D+N -> D
        for i in range(num_layers):#将前面定义的ModuleList进行添加
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))
    
    def _collate_adjacency(self, a, sparse=True):
        #将动态邻接矩阵转换为稀疏形式，以用于图神经网络的计算
        i_list = []
        v_list = []
        #a：动态邻接矩阵，其形状为(batchsize, num_timepoints, num_nodes, num_nodes)
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                #函数通过计算邻接矩阵的阈值来确定稀疏性。使用np.percentile函数找到第(100 - sparsity)
                #百分位数的值，然后通过比较邻接矩阵中的元素与这个阈值，来生成一个二值化的邻接矩阵（thresholded_a），
                # 其中高于阈值的元素被设置为1，其余为0。
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100-self.sparsity))
                #找到thresholded_a中非零元素的索引，_i是一个二维张量，第一维是行索引，第二维是列索引
                _i = thresholded_a.nonzero(as_tuple=False)
                #为非零元素创建一个值列表，所有非零元素的值都被设置为1
                _v = torch.ones(len(_i))
                #调整索引，使其反映在整体稀疏矩阵中的位置。这里通过加上偏移量来实现，偏移量是样本和时间点的累积乘积。
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        #将所有时间点和样本的索引和值列表合并，并转换为稀疏张量所需的格式。.T表示转置索引，.to(a.device)确保索引和值与邻接矩阵在同一个设备上（CPU或GPU）
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        #创建并返回一个稀疏的FloatTensor，其形状为(batch_size * num_timepoints * num_nodes, batch_size * num_timepoints * num_nodes)。
        # 这里使用了_i作为索引，_v作为值，以及邻接矩阵的维度来定义稀疏张量的形状
        return torch.sparse_coo_tensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))
    
    def forward(self, dyn_t, dyn_a, e_s, X_enc, sampling_endpoints):
        logit = 0.0

        # temporal attention
        attention_list = []
        h_dyn_list = []
        minibatch_size, num_timepoints, num_nodes = dyn_a.shape[:3]

        h = rearrange(dyn_t, 'b t n c -> (b t n) c') 
        h = self.initial_linear(h)#初始化的H矩阵

        dyn_a_ = self._collate_adjacency(dyn_a) # a: b t v v
        for layer, (G, T, L) in enumerate(zip(self.gnn_layers, self.transformer_modules, self.linear_layers)):
            # graph convolution
            h = G(h, dyn_a_)
            h_bridge = rearrange(h, '(b t n) c -> b t n c', t=num_timepoints, b=minibatch_size, n=num_nodes) # H-> b t n c

            # spatial attention readout
            q_s = repeat(e_s, 'b c -> t b c', t=num_timepoints) + X_enc[[p-1 for p in sampling_endpoints]] #原本：e_s -> b c  X_enc->  t b c
            q_s = repeat(q_s, "t b c -> b t c l", l=1) # b t c 1
            SA_t = torch.softmax(torch.matmul(h_bridge, q_s), axis=2) + 1
            # SA_t = 1.0
            
            X_dec = h_bridge*SA_t # b t n c
            X_dec = X_dec.sum(axis=2) # b t c
            X_dec = rearrange(X_dec, 'b t c -> t b c')

            # time attention
            h_attend, time_attn = T(X_dec, X_enc, X_enc) #h_attend: t b c
            h_dyn = self.cls_token(h_attend) #h_dyn: b c
            logit += self.dropout(L(h_dyn))

            attention_list.append(time_attn)

        tatt = torch.stack(attention_list, dim=1)
        time_attention = tatt.detach().cpu() #b 4 t t

        return logit, time_attention


class SubtaskLayer(nn.Module):
    def __init__(self, n_region, hidden_dim, subtask_nums):
        super().__init__()
        # @ For X_enc
        # self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        #                          nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

        self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim),nn.LayerNorm(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),nn.LayerNorm(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=1, dropout=0.1)
        self.subtask_linear = nn.Linear(hidden_dim, subtask_nums)

    def forward(self, t):
        timepoints, batchsize = t.shape[:2]
        X_enc = rearrange(t, 't b c -> (b t) c')  # t -> timepoints  b->batchsize  c->?
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=timepoints, b=batchsize)  # 先合并，MLP后再展开  X_enc B*T*N->B*T*C
        X_enc, (hn, cn) = self.PE(X_enc)  # 不改变X_enc维度
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc)  # t b c
        subtask_logit = self.subtask_linear(rearrange(X_enc, 't b c -> (b t) c'))

        return subtask_logit


class SubtaskPredictor(nn.Module):
    def __init__(self, n_region, hidden_dim,subtask_nums_dict):
        super().__init__()

        self.SubPred_dict={'EMOTION': 0, 'GAMBLING': 1, 'LANGUAGE': 2, 'MOTOR': 3, 'RELATIONAL': 4, 'SOCIAL': 5, 'WM': 6}
        self.SubPred_dict_reverse = {0: 'EMOTION', 1: 'GAMBLING', 2: 'LANGUAGE', 3: 'MOTOR', 4: 'RELATIONAL', 5: 'SOCIAL',
                             6: 'WM'}
        self.SubPred_EMOTION=SubtaskLayer(n_region,hidden_dim,subtask_nums_dict['EMOTION'])
        self.SubPred_GAMBLING = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['GAMBLING'])
        self.SubPred_LANGUAGE = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['LANGUAGE'])
        self.SubPred_MOTOR = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['MOTOR'])
        self.SubPred_RELATIONAL = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['RELATIONAL'])
        self.SubPred_SOCIAL = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['SOCIAL'])
        self.SubPred_WM = SubtaskLayer(n_region, hidden_dim, subtask_nums_dict['WM'])


    def forward(self, t, label):
        if(label=='EMOTION'):
            PredModel=self.SubPred_EMOTION
        elif (label == 'GAMBLING'):
            PredModel = self.SubPred_GAMBLING
        elif (label == 'LANGUAGE'):
            PredModel = self.SubPred_LANGUAGE
        elif (label == 'MOTOR'):
            PredModel = self.SubPred_MOTOR
        elif (label == 'RELATIONAL'):
            PredModel = self.SubPred_RELATIONAL
        elif (label == 'SOCIAL'):
            PredModel = self.SubPred_SOCIAL
        elif (label == 'WM'):
            PredModel = self.SubPred_WM

        subtask_logit=PredModel(t)

        return subtask_logit

class MaintaskPredictor(nn.Module):
    def __init__(self, n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.sparsity = sparsity

        self.encoder = BrainEncoder(n_region, hidden_dim)
        self.decoder = BrainDecoder(n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size,
                                    dropout)

        # subtask_nums=max(subtask_nums_list)
        # self.subtaskPred = [SubtaskPredictor(n_region,hidden_dim,subtask_nums) for iindex in range(num_classes)]

    def forward(self, dyn_t, dyn_a, t, a, sampling_endpoints):
        X_enc, e_s = self.encoder(t, a)
        logit, time_attention = self.decoder(dyn_t, dyn_a, e_s, X_enc, sampling_endpoints)
        # pred = logit.argmax(1)  # main task 预测值
        # subtask_logit=self.subtaskPred[pred](t)

        return logit, time_attention  # main task, subtask, time attention

# class BrainNetFormer(nn.Module):
#     def __init__(self, n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size, subtask_nums_list, dropout=0.5):
#         super().__init__()
#         self.num_classes = num_classes
#         self.sparsity = sparsity
#
#         self.encoder = BrainEncoder(n_region, hidden_dim)
#         self.decoder = BrainDecoder(n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size,
#                                     dropout)
#
#         subtask_nums=max(subtask_nums_list)
#
#         self.subtaskPred = [SubtaskPredictor(n_region,hidden_dim,subtask_nums) for iindex in range(num_classes)]
#
#     def forward(self, dyn_t, dyn_a, t, a, sampling_endpoints):
#         X_enc, e_s = self.encoder(t, a)
#         logit, time_attention = self.decoder(dyn_t, dyn_a, e_s, X_enc, sampling_endpoints)
#         pred = logit.argmax(1)  # main task 预测值
#         subtask_logit=self.subtaskPred[pred](t)
#
#         return logit, subtask_logit, time_attention  # main task, subtask, time attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


