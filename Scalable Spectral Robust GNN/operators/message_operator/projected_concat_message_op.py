import torch
import torch.nn.functional as F

from torch.nn import ModuleList
from operators.base_operator import MessageOp
from operators.utils import squeeze_first_dimension
from models.base_scalable.simple_models import MultiLayerPerceptron  


class ProjectedConcatMessageOp(MessageOp):
    def __init__(self, start, end, feat_dim, hidden_dim, num_layers, dropout):
        super(ProjectedConcatMessageOp, self).__init__(start, end)
        self.aggr_type = "proj_concat"

        self.learnable_weight = ModuleList()
        for _ in range(end - start):
            self.learnable_weight.append(MultiLayerPerceptron(
                feat_dim, hidden_dim, num_layers, hidden_dim, dropout))

    def combine(self, feat_list):
        feat_list = squeeze_first_dimension(feat_list)
        adopted_feat_list = feat_list[self.start:self.end]

        concat_feat = self.learnable_weight[0](adopted_feat_list[0])
        for i in range(1, self.end - self.start):
            transformed_feat = F.relu(
                self.learnable_weight[i](adopted_feat_list[i]))
            concat_feat = torch.hstack((concat_feat, transformed_feat))

        return concat_feat
