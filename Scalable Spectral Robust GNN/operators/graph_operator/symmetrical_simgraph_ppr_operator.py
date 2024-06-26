import scipy.sparse as sp

from operators.base_operator import GraphOp
from operators.utils import adj_to_symmetric_norm


class PprGraphOp(GraphOp):
    def __init__(self, prop_steps, r=0.5, alpha=0.15):
        super(PprGraphOp, self).__init__(prop_steps)
        self.r = r
        self.alpha = alpha

    def construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.r)
        adj_normalized = (1 - self.alpha) * adj_normalized + self.alpha * sp.eye(adj.shape[0])
        return adj_normalized.tocsr()
