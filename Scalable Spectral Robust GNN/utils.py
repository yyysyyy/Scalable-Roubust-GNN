import torch
import numpy as np
import scipy.sparse as sp
import numpy.ctypeslib as ctl
import os.path as osp
import random
from ctypes import c_int
from scipy.sparse import coo_matrix

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def generate_numbers(n, i, numbers):
    numbers.remove(i)
    selected_numbers = random.sample(numbers, n)
    numbers.append(i)
    return selected_numbers

def compute_distance(candiate_features, node_feature):
    repeat_feature = node_feature.repeat( candiate_features.shape[0], 1)
    distance =  torch.norm(repeat_feature-candiate_features, dim = 1)
    return distance
