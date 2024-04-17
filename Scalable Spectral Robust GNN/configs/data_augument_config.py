import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_name', help='unsigned & undirected & unweighted simplex homogeneous sparsity data name', type=str, default="pubmed_0.6_0.6")
#parser.add_argument('--train_model_path', help='Model path for handling sparse node features', type=str, default="./models/train_model/cora.pt")
parser.add_argument('--data_root', help='unsigned & undirected & unweighted simplex homogeneous data root', type=str, default="./sparsity_datasets/simhomo/Planetoid")
parser.add_argument('--data_save_path', help='the path to save augument dataset', type=str, default="./augument_datasets/simhomo/Planetoid/")
parser.add_argument('--data_split', help='unsigned & undirected & unweighted simplex homogeneous data split method', type=str, default="official")
parser.add_argument('--dropout', help='drop out of gnn model', type=float, default=0.5)
parser.add_argument('--weight_decay', help='weight decay', type=float, default=5e-4)
parser.add_argument('--hidden_dim', help='hidden units of gnn model', type=int, default=256)
parser.add_argument('--num_layers', help='the num of the layers', type=int, default=3)
parser.add_argument('--batch_size', help='the num of the batch', type=int, default=300)
parser.add_argument('--prop_steps', help='prop steps', type=int, default=3)
parser.add_argument('--r', help='symmetric normalized unit', type=float, default=0.5)
parser.add_argument('--degree_level', help='the minimum number of degrees each node should have', type=int, default=1)
data_augument_args = parser.parse_args()
