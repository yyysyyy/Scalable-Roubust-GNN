from models.base_scalable.base_model import SpectralModel
from models.base_scalable.simple_models import MultiLayerPerceptron, Wavelet2NeuralNetwork
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymLaplacianGraphOp
from operators.message_operator.last_message_op import LastMessageOp

class WAVELAT(SpectralModel):
    def __init__(self, ncount, scale, approximation_order, tolerance, feat_dim, hidden_dim,output_dim, dropout):
        super(WAVELAT, self).__init__(scale, approximation_order, tolerance)
        self.pre_graph_op = None
        self.pre_msg_op = None
        self.base_model = Wavelet2NeuralNetwork(feat_dim, hidden_dim, output_dim, dropout,ncount)
