import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import ClusterData, ClusterLoader
import torch_geometric.transforms as T
import torch_geometric
from utils import sgc_precompute
from model import SGC
from ogb.nodeproppred import PygNodePropPredDataset
from sparsity_dataset import SparsityDataset

torch.manual_seed(12345)
dataset = SparsityDataset(root='sparsity_dataset', name='ogbn-products')
data = dataset[0]
cluster_data = ClusterData(data, num_parts=128)
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
total_num_nodes = 0
for step, sub_data in enumerate(train_loader):
    sub_data = sub_data.to('cuda')
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    sum = sub_data.train_mask.sum()
    print(f'Number of training nodes in the current batch: {sum}')
    print()
    total_num_nodes += sub_data.num_nodes

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')
