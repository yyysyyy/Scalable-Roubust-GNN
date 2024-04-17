import torch
import torch_geometric.transforms as T
import torch_geometric
import numpy
import os
import utils
from datasets.simhomo.planetoid import Planetoid
from datasets.simhomo.ogbn import Ogbn
from configs.data_process_config import data_process_args
from torch_geometric.utils import remove_self_loops

def dataRead(root, dataset):
    #cora, citeseer, pubmed, ogb-arxiv, ogb-products, ogb-paper100m, reddit, flickr
    if dataset.lower() in ('cora', 'citeseer', 'pubmed'):
        data = Planetoid(dataset, root, 'official')
    elif dataset.lower() in ('arxiv', 'products'):
        data = Ogbn(dataset, root, 'official')
    elif dataset == 'reddit':
        data = torch_geometric.datasets.Reddit(root=(root + '/reddit'))
        data = data[0]
        data.train_idx = data.train_mask.nonzero().view(-1)
        data.val_idx = data.val_mask.nonzero().view(-1)
        data.test_idx = data.test_mask.nonzero().view(-1)
    elif dataset == 'flickr':
        data = torch_geometric.datasets.Flickr(root=(root + '/flickr'))
        data = data[0]
        data.train_idx = data.train_mask.nonzero().view(-1)
        data.val_idx = data.val_mask.nonzero().view(-1)
        data.test_idx = data.test_mask.nonzero().view(-1)
    else:
        print('Error: dataset not found!')
        return
    return data

def featureMasked(data, r):    
    if data.x.dtype == torch.float32:
        data.x = data.x.numpy()
    mask = torch.rand(data.x.shape)
    mask = (mask > r).int()
    feature = torch.from_numpy(data.x).float()
    return mask, feature

def edgeMasked(data, shading_rate, data_name):
    if data_name == 'reddit' or data_name == 'flickr':
        edge_index = data.edge_index
        row, col = edge_index[0], edge_index[1]
    else:
        adj = data.adj.tocoo()
        row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)
        edge_index = torch.stack([row, col], dim=0)   
    #return edge_index
    dir_mask = col > row
    edge_index = edge_index[:, dir_mask]
    
    need_delete_num = int(edge_index.shape[1] * shading_rate)
    delete_num = 0
    i = 0
    mask = data.y[edge_index[0]] != data.y[edge_index[1]]
    # while i < edge_index.shape[1] and delete_num < need_delete_num:
    #     if data.y[edge_index[0][i]] != data.y[edge_index[1][i]]:
    #         edge_index = torch.cat((edge_index[:, :i], edge_index[:, i+1:]), dim=1)
    #         i -= 1
    #         delete_num += 1
    #     i += 1
    mask = torch.randperm(edge_index.shape[1])[need_delete_num-delete_num:]
    edge_index = edge_index[:,mask]
    return mask, edge_index

def dataSave(feature, edge_index, feature_mask, edge_mask, data, args):
    if args.dataset == 'arxiv' or args.dataset == 'products' or args.dataset == 'papers100M':
        path = 'sparsity_datasets/simhomo/ogbn/' + args.dataset + '_' + str(args.sparse_rate[0]) + '_' + str(args.sparse_rate[1]) + '/raw/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
    elif args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        path = 'sparsity_datasets/simhomo/Planetoid/' + args.dataset + '_' + str(args.sparse_rate[0]) + '_' + str(args.sparse_rate[1]) + '/raw/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
    elif args.dataset == 'reddit':
        path = 'sparsity_datasets/simhomo/reddit/reddit_' + str(args.sparse_rate[0]) + '_' + str(args.sparse_rate[1]) + '/raw/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
    elif args.dataset == 'flickr':
        path = 'sparsity_datasets/simhomo/flickr/flickr_' + str(args.sparse_rate[0]) + '_' + str(args.sparse_rate[1]) + '/raw/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
    #path = 'sparsity_dataset/' + args.dataset + '/raw/'
    feature_file = path + 'feature.pt'
    edge_index_file = path + 'edge_index.pt'
    label_file = path + 'label.pt'
    train_idx_file = path + 'train_idx.pt'
    val_idx_file = path + 'val_idx.pt'
    test_idx_file = path + 'test_idx.pt'
    feature_mask_file = path + 'feature_mask.pt'
    edge_mask_file = path + 'edge_mask.pt'
    with open(feature_file, 'wb') as f:
        torch.save(feature, f)
    with open(edge_index_file, 'wb') as f:
        torch.save(edge_index, f)
    with open(label_file, 'wb') as f:
        torch.save(data.y, f)
    with open(train_idx_file, 'wb') as f:
        torch.save(data.train_idx, f)
    with open(val_idx_file, 'wb') as f:
        torch.save(data.val_idx, f)
    with open(test_idx_file, 'wb') as f:
        torch.save(data.test_idx, f)
    with open(feature_mask_file, 'wb') as f:
        torch.save(feature_mask, f)
    with open(edge_mask_file, 'wb') as f:
        torch.save(edge_mask, f)

if __name__ == '__main__':
    
    utils.seed_everything(data_process_args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataRead(data_process_args.dataroot, data_process_args.dataset)
    sparsity_rate = data_process_args.sparse_rate
    feature_mask, feature = featureMasked(data, sparsity_rate[0])
    edge_mask, edge_index = edgeMasked(data, sparsity_rate[1], data_process_args.dataset)
    dataSave(feature, edge_index, feature_mask, edge_mask, data, data_process_args)
    print('done')