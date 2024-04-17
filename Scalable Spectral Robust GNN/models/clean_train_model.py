from models.base_scalable.base_model import FeatureAugumentModel
from models.base_scalable.simple_models import FeatureAugument2MLP
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.last_message_op import LastMessageOp

class CleanTrainModel(FeatureAugumentModel):
    def __init__(self, prop_steps, r, feat_dim, hidden_dim,output_dim,dropout=0.0):
        super(CleanTrainModel, self).__init__(prop_steps, feat_dim, output_dim)
        #self.pre_graph_op = SymLaplacianGraphOp(prop_steps, r=r)
        #self.pre_msg_op = LastMessageOp()
        self.base_model = FeatureAugument2MLP(feat_dim, hidden_dim, output_dim,dropout)
