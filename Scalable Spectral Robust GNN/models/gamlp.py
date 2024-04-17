from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import MultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.learnable_weighted_messahe_op import LearnableWeightedMessageOp

class GAMLP(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim, hidden_dim, num_layers, dropout):
        super(GAMLP, self).__init__(prop_steps, feat_dim, output_dim)

        self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.pre_msg_op = LearnableWeightedMessageOp(0, prop_steps + 1, "jk", prop_steps, feat_dim)
        self.base_model = MultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim, dropout)