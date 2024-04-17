from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import LogisticRegression
from operators.message_operator.mean_message_op import MeanMessageOp 
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
   

class SSGC(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim):
        super(SSGC, self).__init__(prop_steps, feat_dim, output_dim)

        self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.pre_msg_op = MeanMessageOp(start=0, end=prop_steps + 1)
        self.base_model = LogisticRegression(feat_dim, output_dim)