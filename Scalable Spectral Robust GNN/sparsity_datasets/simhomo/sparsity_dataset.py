import torch
import numpy as np
import os.path as osp
import pickle as pkl
import pandas as pd

from datasets.base_data import Graph
from configs.data_config import data_args
from datasets.base_dataset import NodeDataset
from ogb.nodeproppred import PygNodePropPredDataset
from datasets.utils import pkl_read_file, remove_self_loops, to_undirected, edge_homophily, node_homophily, linkx_homophily, set_spectral_adjacency_reg_features

class Sparsity_Dataset(NodeDataset):
    '''
    Dataset description: (Open Graph Benchmark): https://ogb.stanford.edu/docs/nodeprop/
    Directed infomation:    Undirected network (ogbn-products)
                            Directed network (ogbn-arxiv) -> we implement it as an undirected graph.

    -> ogbn-arxiv:     unsigned & undirected & unweighted homogeneous simplex network    
    -> ogbn-products:  unsigned & undirected & unweighted homogeneous simplex network

    We remove all multiple edges and self-loops from the original dataset. The above phenomenon result in a different number of edges compared to the original report -> NeurIPS'21 Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, LINKX https://arxiv.org/pdf/2110.14446.pdf
    -> Multiple edges do not affect the number of edges, but may lead to the aggregation of edge weights.
    -> ogbn-arxiv: (1,166,243 -> 1,157,799)
    -> ogbn-products: (61,859,140 -> 61,859,012)
    
    ogbn-arxiv:     The ogbn-arxiv dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers.
                    169,343 nodes, 1,157,799 edges, 128 feature dimensions, 40 classes num.
                    Edge homophily: 0.6542, Node homophily:0.6353, Linkx homophily:0.4211.

    ogbn-products:  The ogbn-products dataset is an undirected and unweighted graph, representing an Amazon product co-purchasing network. 
                    Nodes represent products sold in Amazon, and edges between two products indicate that the products are purchased together. 
                    2,449,029 nodes, 61,859,012 edges, 100 feature dimensions, 47 classes num.
                    Edge homophily: 0.8076, Node homophily:0.833, Linkx homophily:0.4591.
    
    split:
        ogbn-arxiv:
            official:   We propose to train on papers published until 2017, 
                        validate on those published in 2018, 
                        and test on those published since 2019.
                        train/val/test = 90,941/29,799/48,603

        ogbn-products:
            official:   We sort the products according to their sales ranking 
                        and use the top 8% for training, 
                        next top 2% for validation, 
                        and the rest for testing. 
    '''
    def __init__(self, name="arxiv_0_0_0_0", root="./sparsity_datasets/simhomo/ogbn", split="official", k=None, is_augumented=False):
        super(Sparsity_Dataset, self).__init__(root, name, k, is_augumented)
        self.raw_paths = self.raw_file_paths
        self.read_file()
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(split)
        self.num_node_classes = self.num_classes
        self.num_edge_classes = None
        self.is_augumented = is_augumented
        self.edge_homophily = edge_homophily(self.adj, self.y)
        self.node_homophily = node_homophily(self.adj, self.y)
        self.linkx_homophily = linkx_homophily(self.adj, self.y)
        
    @property
    def raw_file_paths(self):
        filenames = ['feature.pt', 'edge_index.pt', 'label.pt', 'train_idx.pt', 'val_idx.pt', 'test_idx.pt', 'feature_mask.pt','edge_mask.pt']
        return [osp.join(self.raw_dir,filename) for filename in filenames]

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self.processed_dir, "{}.{}".format(self.name, filename))

    @property
    def raw_file_names(self):
        return ['feature.pt', 'edge_index.pt', 'label.pt', 'train_idx.pt', 'val_idx.pt', 'test_idx.pt', 'feature_mask.pt','edge_mask.pt']
    
    def _get_features(self):
        return torch.load(self.raw_paths[0])
    
    def _get_edge_index(self):
        return torch.load(self.raw_paths[1])
    
    def _get_labels(self):
        return torch.load(self.raw_paths[2])
    
    def _get_train_idx(self):
        return torch.load(self.raw_paths[3])
    
    def _get_val_idx(self):
        return torch.load(self.raw_paths[4])
    
    def _get_test_idx(self):
        return torch.load(self.raw_paths[5])
    
    def _get_feature_mask(self):
        return torch.load(self.raw_paths[6])
    
    def _get_edge_mask(self):
        return torch.load(self.raw_paths[7])
    
    def read_file(self):
        self.data = pkl_read_file(self.processed_file_paths)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        #self.masked_adj = self.data.masked_adj
        self.edge_type = self.data.edge_type
        self.num_features = self.data.num_features
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge
        if self.is_augumented:
            self.feature_mask = None
            self.edge_mask = None
        else:
            self.feature_mask = self.data.feature_mask
            self.edge_mask = self.data.edge_mask
        import time
        t1 = time.time()
        edge_weight = self.edge.edge_weight
        indices = torch.vstack((self.edge.row, self.edge.col)).long()
        edge_num_node = indices.max().item() + 1
        features = set_spectral_adjacency_reg_features(edge_num_node, indices, edge_weight)

    def download(self):
        pass

    def process(self):
        # dataset = PygNodePropPredDataset("ogbn-" + self.name, self.raw_dir)

        # data = dataset[0]
        # features, labels = data.x.numpy().astype(np.float32), data.y.to(torch.long).squeeze(1)
        # num_node = data.num_nodes

        # if self.name == "arxiv":
        #     undi_edge_index = torch.unique(data.edge_index, dim=1)
        #     undi_edge_index = to_undirected(undi_edge_index)
        # elif self.name == "products":
        #     undi_edge_index = data.edge_index
        # undi_edge_index = torch.unique(undi_edge_index, dim=1)
        # undi_edge_index = remove_self_loops(undi_edge_index)[0]
        
        # row, col = undi_edge_index
        # edge_weight = torch.ones(len(row))
        # edge_type = "UUU"
        self.raw_paths = self.raw_file_paths
        edge_index = self._get_edge_index()
        x = self._get_features()
        x = x.cpu().numpy()
        if self.is_augumented:
            feature_mask = None
            edge_mask = None
        else:
            feature_mask = self._get_feature_mask()
            edge_mask = self._get_edge_mask()
        y = self._get_labels()
        row, col = edge_index
        edge_type = 'UUU'
        num_node = x.shape[0]
        edge_weight = torch.ones(row.shape[0])
        g = Graph(row, col, edge_weight, num_node, edge_type, feature_mask, edge_mask, x=x, y=y)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def generate_split(self, split):
        if split == "official":
            train_idx = self._get_train_idx()
            val_idx = self._get_val_idx()
            test_idx = self._get_test_idx()
        elif split == "random":
            raise NotImplementedError
        
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
