import torch
import numpy as np
from random import randrange


def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:
            fc = corrcoef(_t[i:i+window_size].T)
            if not self_loop: fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))
    return torch.stack(dynamic_fc_list, dim=1), sampling_points

def process_static_fc(timeseries, self_loop=True):
    static_fc_list = []
    for _t in timeseries:
        fc = corrcoef(_t.T)
        if not self_loop: fc -= torch.eye(fc.shape[0])
        static_fc_list.append(fc)
    return torch.stack(static_fc_list, dim=0)

def process_dynamic_t(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_t_list = []
    for i in sampling_points:
        t_list = []
        for _t in timeseries:
            #@ 添加分段时间序列
            t_list.append(_t[i:i+window_size].T)
        dynamic_t_list.append(torch.stack(t_list))
    return torch.stack(dynamic_t_list, dim=1), sampling_points

# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    # print(f"c0:{c}")
    c = c.div(stddev.expand_as(c))
    # print(f"c1:{c}")
    c = c.div(stddev.expand_as(c).t())
    # print(f"c2:{c}")
    c = torch.clamp(c, -1.0, 1.0)
    # print(f"c3:{c}")
    return c
