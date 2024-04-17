from sparsity_datasets.simhomo.sparsity_dataset import Sparsity_Dataset

def load_homo_simplex_sparsity_dataset(name, root, split, is_augumented = False):
    name == name.lower()
    dataset = Sparsity_Dataset(name, root, split, is_augumented=is_augumented)
    #print("Edge homophily: {}, Node homophily:{}, Linkx homophily:{}".format(round(dataset.edge_homophily, 4), round(dataset.node_homophily, 4), round(dataset.linkx_homophily, 4)))
    return dataset
