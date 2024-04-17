import torch
import torch.nn as nn
import numpy
import random
import time
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric
import copy
from ogb.nodeproppred import PygNodePropPredDataset
from model import MLP
from torch_geometric.data import DataLoader
from torch_sparse import SparseTensor
from sparsity_dataset import SparsityDataset

def dataRead(dataroot, dataset):
    # if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
    #     data = torch_geometric.datasets.Planetoid(root=dataroot, name=dataset, transform=T.ToSparseTensor())
    # elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-products' or dataset == 'ogbn-papers100M':
    #     data = PygNodePropPredDataset(root=dataroot, name=dataset, transform=T.ToSparseTensor())
    #     split_idx = data.get_idx_split()
    # elif dataset == 'reddit':
    #     data = torch_geometric.datasets.Reddit(root=(dataroot + '/reddit'), transform=T.ToSparseTensor())
    # elif dataset == 'flickr':
    #     data = torch_geometric.datasets.Flickr(root=(dataroot + '/flickr'), transform=T.ToSparseTensor())
    # else:
    #     print('Error: dataset not found!')
    #     return
    # data = data[0]
    # if dataset == 'ogbn-arxiv' or dataset == 'ogbn-products' or dataset == 'ogbn-papers100M':
    #     data.train_mask = split_idx['train']
    #     data.val_mask = split_idx['valid']
    #     data.test_mask = split_idx['test']
    data = SparsityDataset(root=dataroot, name=dataset, transform=T.ToSparseTensor())
    data = data[0]
    return data

def splitLabels(labels):
    nclass = labels.max().item() + 1
    labels_split = []
    labels_split_dif = []
    for i in range(nclass):
        labels_split.append((labels == i).nonzero().view(-1))
    for i in range(nclass):
        dif_type = [t for t in range(nclass) if t != i]
        labels_dif = torch.cat([labels_split[t] for t in dif_type])
        labels_split_dif.append(labels_dif)
    return labels_split, labels_split_dif

def tripletLoss(n_sample_class, n_class, labels_split, labels_split_dif, logits, thre):
    loss = 0
    for i in range(n_class):
        randIndex1 = random.choices(labels_split[i], k=n_sample_class)
        randIndex2 = random.choices(labels_split[i], k=n_sample_class)
        feats1 = logits[randIndex1]
        feats2 = logits[randIndex2]
        randIndex_dif = random.choices(labels_split_dif[i], k=n_sample_class)
        feats_dif = logits[randIndex_dif]
        dist1 = torch.sum((feats1 - feats2) ** 2, dim=1)
        dist2 = torch.sum((feats1 - feats_dif) ** 2, dim=1)
        loss += torch.sum(F.relu(dist1 - dist2 + thre))
    loss /= n_sample_class*n_class
    return loss

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sgc_precompute(feature, adj, degree):
    t = time.time()
    for i in range(degree):
        feature = adj @ feature
        t2 = time.time()
        print('sgc_precompute time in degree {}: {}'.format(i, t2 - t))
        t = t2
    return feature
