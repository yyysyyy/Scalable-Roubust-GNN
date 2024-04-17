import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed',help='cora, citeseer, pubmed, reddit, flickr, arxiv, products')
parser.add_argument('--dataroot', type=str, default='./datasets/simhomo/',help='choose the dataroot')
parser.add_argument('--seed', help='seed everything', type=int, default=2023)
parser.add_argument('--sparse_rate', type=list, default=[0.6,0.6],help='sparse rate')
data_process_args = parser.parse_args()