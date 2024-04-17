import argparse

parser = argparse.ArgumentParser()


# dataset
# For the local cache split results, we perform 10 random splits for the node-level semi-supervised node classification task and link-level link prediction tasks
# 1. unsigned & undirected & unweighted simplex homogeneous data
    #   name: cora & citeseer & pubmed & arxiv (small), products (medium)
    #   root: ./datasets/simhomo/
    #   split:  official (Random splitting of fixed rates)
parser.add_argument('--data_name', help='unsigned & undirected & unweighted simplex homogeneous data name', type=str, default="arxiv_0.7_0.7")
parser.add_argument('--data_root', help='unsigned & undirected & unweighted simplex homogeneous data root', type=str, default="./augument_datasets/simhomo/ogbn/")
parser.add_argument('--data_split', help='unsigned & undirected & unweighted simplex homogeneous data split method', type=str, default="official")

data_args = parser.parse_args()
