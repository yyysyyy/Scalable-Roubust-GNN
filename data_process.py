import torch
import torch_geometric.transforms as T
import torch_geometric
import argparse
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor

def dataRead(dataroot, dataset):
    #cora, citeseer, pubmed, ogb-arxiv, ogb-products, ogb-paper100m, reddit, flickr
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        data = torch_geometric.datasets.Planetoid(root=dataroot, name=dataset)
    elif dataset == 'ogbn-arxiv' or dataset == 'ogbn-products' or dataset == 'ogbn-papers100M':
        data = PygNodePropPredDataset(root=dataroot, name=dataset, transform=T.ToSparseTensor())
        split_idx = data.get_idx_split()
    elif dataset == 'reddit':
        data = torch_geometric.datasets.Reddit(root=(dataroot + '/reddit'), transform=T.ToSparseTensor())
    elif dataset == 'flickr':
        data = torch_geometric.datasets.Flickr(root=(dataroot + '/flickr'), transform=T.ToSparseTensor())
    else:
        print('Error: dataset not found!')
        return
    data = data[0]
    data.adj_t = data.adj_t.to_symmetric()
    if dataset == 'ogbn-arxiv' or dataset == 'ogbn-products' or dataset == 'ogbn-papers100M':
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[split_idx['valid']] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[split_idx['test']] = 1
    return data

def featureMasked(data, partly_sparse_rate, completely_sparse_rate, shading_rate, args):
    # data 读入的图数据
    # partly_sparse_rate 部分缺失率
    # completely_sparse_rate 完全缺失率（只作用于验证集和测试集）
    train_idx, val_idx, test_idx = data.train_mask, data.val_mask, data.test_mask
    if args.dataset == 'ogbn-arxiv' or args.dataset == 'ogbn-products' or args.dataset == 'ogbn-papers100M':
        train_sparse_idx = torch.randperm(train_idx.shape[0])[:int(train_idx.shape[0] * partly_sparse_rate)]
        val_sparse_idx = torch.randperm(val_idx.shape[0])[:int(val_idx.shape[0] * (partly_sparse_rate + completely_sparse_rate))]
        test_sparse_idx = torch.randperm(test_idx.shape[0])[:int(test_idx.shape[0] * (partly_sparse_rate + completely_sparse_rate))]
    else:
        train_sparse_idx = torch.randperm(train_idx.sum())[:int(train_idx.sum() * partly_sparse_rate)]
        val_sparse_idx = torch.randperm(val_idx.sum())[:int(val_idx.sum() * (partly_sparse_rate + completely_sparse_rate))]
        test_sparse_idx = torch.randperm(test_idx.sum())[:int(test_idx.sum() * (partly_sparse_rate + completely_sparse_rate))]
    mask = torch.ones_like(data.x, dtype=torch.bool)
    x = torch.ones_like(mask[val_idx], dtype=torch.bool)
    x[val_sparse_idx[:int(val_idx.sum() * completely_sparse_rate)]] = 0
    mask[val_idx] = x
    x = torch.ones_like(mask[test_idx], dtype=torch.bool)
    x[test_sparse_idx[:int(test_idx.sum() * completely_sparse_rate)]] = 0
    mask[test_idx] = x
    #mask[test_idx][test_sparse_idx[:int(test_idx.sum() * completely_sparse_rate)]] = 0
    for i in range(len(train_sparse_idx)):
        mask[train_sparse_idx[i],torch.randperm(data.x.shape[1])[:int(data.x.shape[1] * shading_rate)]] = 0
    for i in range(int(val_sparse_idx.sum() * completely_sparse_rate), len(val_sparse_idx)):
        mask[val_sparse_idx[i]][torch.randperm(data.x.shape[1])[:int(data.x.shape[1] * shading_rate)]] = 0
    for i in range(int(test_sparse_idx.sum() * completely_sparse_rate), len(test_sparse_idx)):
        mask[test_sparse_idx[i]][torch.randperm(data.x.shape[1])[:int(data.x.shape[1] * shading_rate)]] = 0
    feature = torch.mul(data.x, mask)
    return feature

def edgeMasked(data, shading_rate):
    row, col, edge_attr = data.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    mask = torch.randperm(edge_index.shape[1])[int(edge_index.shape[1] * shading_rate):]
    edge_index = edge_index[:,mask]
    return edge_index

def dataSave(feature, edge_index, data, args):
    path = 'sparsity_dataset/' + args.dataset + '/raw/'
    feature_file = path + 'feature.pt'
    edge_index_file = path + 'edge_index.pt'
    label_file = path + 'label.pt'
    train_mask_file = path + 'train_mask.pt'
    val_mask_file = path + 'val_mask.pt'
    test_mask_file = path + 'test_mask.pt'
    with open(feature_file, 'wb') as f:
        torch.save(feature, f)
    with open(edge_index_file, 'wb') as f:
        torch.save(edge_index, f)
    with open(label_file, 'wb') as f:
        torch.save(data.y, f)
    with open(train_mask_file, 'wb') as f:
        torch.save(data.train_mask, f)
    with open(val_mask_file, 'wb') as f:
        torch.save(data.val_mask, f)
    with open(test_mask_file, 'wb') as f:
        torch.save(data.test_mask, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',help='cora, citeseer, pubmed, reddit, flickr, ogbn-arxiv, ogbn-products, ogbn-papers100M')
    parser.add_argument('--dataroot', type=str, default='dataset',help='choose the dataroot')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataRead(args.dataroot, args.dataset)
    feature = featureMasked(data, 0, 0, 0, args)
    edge_index = edgeMasked(data, 0)
    dataSave(feature, edge_index, data, args)