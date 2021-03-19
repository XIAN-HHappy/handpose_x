#-*-coding:utf-8-*-
# date:2020-04-11
# Author: Eric.Lee
# function: model utils

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / float(total)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_seed(seed = 666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
