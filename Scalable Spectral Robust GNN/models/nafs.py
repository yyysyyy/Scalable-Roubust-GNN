from models.base_scalable.base_model import BaseSGModel
from models.base_scalable.simple_models import LogisticRegression
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.over_smooth_distance_op import OverSmoothDistanceWeightedOp
 
class NAFS(BaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim):
        super(NAFS, self).__init__(prop_steps, feat_dim, output_dim)

        self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        self.pre_msg_op = OverSmoothDistanceWeightedOp()
        self.base_model = LogisticRegression(feat_dim, output_dim)