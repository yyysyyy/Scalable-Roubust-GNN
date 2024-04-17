import os.path as osp
import torch
import numpy
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_sparse import SparseTensor

class SparsityDataset(Dataset):
    def __init__(self, root='./sparsity_dataset', name = 'cora',transform=None, pre_transform=None, pre_filter=None, use_edge_attr = False):
        self.root = root
        self.name = name
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root + '/' + self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root + '/' + self.name, 'processed')
    
    @property
    def raw_file_names(self):
        return ['feature.pt', 'edge_index.pt', 'label.pt', 'train_mask.pt', 'val_mask.pt', 'test_mask.pt']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def _get_features(self):
        return torch.load(self.raw_paths[0])
    
    def _get_edge_index(self):
        return torch.load(self.raw_paths[1])
    
    def _get_labels(self):
        return torch.load(self.raw_paths[2])
    
    def _get_train_mask(self):
        return torch.load(self.raw_paths[3])
    
    def _get_val_mask(self):
        return torch.load(self.raw_paths[4])
    
    def _get_test_mask(self):
        return torch.load(self.raw_paths[5])
    
    def process(self):
        # edge_index = self._get_edge_index()
        # x = self._get_features()
        # y = self._get_labels()
        # train_mask = self._get_train_mask()
        # val_mask = self._get_val_mask()
        # test_mask = self._get_test_mask()
        # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1])
        # adj_t = adj_t.to_symmetric()
        # self.data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, adj_t = adj_t)
        # if self.pre_filter is not None:
        #     self.data = self.pre_filter(self.data)
        # if self.pre_transform is not None:
        #     self.data = self.pre_transform(self.data)
        # torch.save(self.data, self.processed_paths[0])
        edge_index = self._get_edge_index()
        x = self._get_features()
        x = x.cpu().numpy()
        y = self._get_labels()
        train_mask = self._get_train_mask()
        val_mask = self._get_val_mask()
        test_mask = self._get_test_mask()
        row, col = edge_index
        edge_type = 'UUU'
        num_node = x.shape[0]
        edge_weight = torch.ones(row.shape[0])
        return
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data
