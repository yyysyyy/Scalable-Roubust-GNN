import torch
import torch.nn as nn
from torch_sparse import spspmm, spmm
from operators.utils import squeeze_first_dimension

class OneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps, feat_dim):
        super(OneDimConvolution, self).__init__()
        self.adj = None
        self.hop_num = prop_steps
        self.learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            self.learnable_weight.append(nn.Parameter(
                torch.FloatTensor(feat_dim, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class OneDimConvolutionWeightSharedAcrossFeatures(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(OneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
        self.adj = None
        self.hop_num = prop_steps
        self.learnable_weight = nn.ParameterList()
        for _ in range(prop_steps):
            # To help xvarient_uniform_ calculate fan in and fan out, "1" should be kept here.
            self.learnable_weight.append(nn.Parameter(
                torch.FloatTensor(1, num_subgraphs)))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.learnable_weight:
            nn.init.xavier_uniform_(weight)

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (
                    adopted_feat * (self.learnable_weight[i])).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class FastOneDimConvolution(nn.Module):
    def __init__(self, num_subgraphs, prop_steps):
        super(FastOneDimConvolution, self).__init__()
        self.adj = None
        self.num_subgraphs = num_subgraphs
        self.prop_steps = prop_steps

        # How to initialize the weight is extremely important.
        # Pure xavier will lead to extremely unstable accuracy.
        # Initialized with ones will not perform as good as this one.        
        self.learnable_weight = nn.Parameter(
            torch.ones(num_subgraphs * prop_steps, 1))

    # feat_list_list: 3-d tensor (num_node, feat_dim, num_subgraphs * prop_steps)
    def forward(self, feat_list_list):
        return (feat_list_list @ self.learnable_weight).squeeze(dim=2)

    def subgraph_weight(self):
        return self.learnable_weight.view(
            self.num_subgraphs, self.prop_steps).sum(dim=1)


class IdenticalMapping(nn.Module):
    def __init__(self) -> None:
        super(IdenticalMapping, self).__init__()
        self.adj = None

    def forward(self, feature):
        return feature


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.adj = None
        self.query_edges = None
        self.fc_node_edge = nn.Linear(feat_dim, output_dim)
        self.linear = nn.Linear(2*output_dim, output_dim)

    def forward(self, feature):
        feature = squeeze_first_dimension(feature)
        if  self.query_edges == None:
            output = self.fc_node_edge(feature)
        else:
            x = self.fc_node_edge(feature)
            x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, num_layers, dropout, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        self.adj = None
        self.query_edges = None
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.num_layers = num_layers
        self.linear = nn.Linear(2*hidden_dim, output_dim)
        self.fcs_node_edge = nn.ModuleList()
        self.fcs_node_edge.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs_node_edge.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs_node_edge.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs_node_edge:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):

        for i in range(self.num_layers - 1):
            feature = self.fcs_node_edge[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        if  self.query_edges == None:
            output = self.fcs_node_edge[-1](feature)
            
        else:
            x = torch.cat((feature[self.query_edges[:, 0]], feature[self.query_edges[:, 1]]), dim=-1)
            x = self.dropout(x)
            output = self.linear(x)
            
        return output


class ResMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
        super(ResMultiLayerPerceptron, self).__init__()
        self.adj = None
        self.query_edges = None
        if num_layers < 2:
            raise ValueError("ResMLP must have at least two layers!")
        self.num_layers = num_layers
        self.linear = nn.Linear(2*hidden_dim, output_dim)
        self.fcs_node_edge = nn.ModuleList()
        self.fcs_node_edge.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs_node_edge.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs_node_edge.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, feature):
        feature = self.dropout(feature)
        feature = self.fcs_node_edge[0](feature)
        if self.bn is True:
            feature = self.bns[0](feature)
        feature = self.relu(feature)
        residual = feature

        for i in range(1, self.num_layers - 1):
            feature = self.dropout(feature)
            feature = self.fcs_node_edge[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature_ = self.relu(feature)
            feature = feature_ + residual
            residual = feature_
        feature = self.dropout(feature)
        if  self.query_edges == None:
            output = self.fcs_node_edge[-1](feature)
        else:
            x = torch.cat((feature[self.query_edges[:, 0]], feature[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)
        return output


class Layer2GraphConvolution(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(Layer2GraphConvolution, self).__init__()
        self.adj = None
        self.query_edges = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1_node_edge = nn.Linear(feat_dim, hidden_dim)
        self.fc2_node = nn.Linear(hidden_dim, output_dim)
        self.fc2_edge = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, feature):
        x = feature
        x = self.fc1_node_edge(x)
        x = torch.mm(self.adj, x)
        x = self.relu(x)
        x = self.dropout(x)
        if  self.query_edges == None:
            x = self.fc2_node(x)
            output = torch.mm(self.adj, x) 
        else:
            x = self.fc2_edge(x)
            x = torch.mm(self.adj, x) 
            x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
            output = self.linear(x)  
        return output
    
class FeatureAugument2MLP(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(FeatureAugument2MLP, self).__init__()
        self.adj = None
        self.query_edges = None
        self.dropout = nn.Dropout(dropout)
        self.fc1_node_edge = nn.Linear(feat_dim, hidden_dim)
        self.fc2_node = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, feature):
        x = feature
        x = self.fc1_node_edge(x)
        x = self.relu(x)
        x1 = self.dropout(x)
        output = self.fc2_node(x1)
        return x, output


class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """
    def __init__(self, in_channels, out_channels, ncount):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        #self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.ncount)],
                                                         [node for node in range(self.ncount)]])

        #self.diagonal_weight_indices = self.diagonal_weight_indices.to(self.device)
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)

class SparseGraphWaveletLayer(GraphWaveletLayer):
    """
    Sparse Graph Wavelet Layer Class.
    """
    def forward(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values, dropout, device):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """
        self.diagonal_weight_indices = self.diagonal_weight_indices.to(device)
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = spmm(feature_indices,
                                 feature_values,
                                 self.ncount,
                                 self.in_channels,
                                 self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        dropout_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features),
                                                       training=self.training,
                                                       p=dropout)
        return dropout_features

class DenseGraphWaveletLayer(GraphWaveletLayer):
    """
    Dense Graph Wavelet Layer Class.
    """
    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, features, device):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """
        self.diagonal_weight_indices = self.diagonal_weight_indices.to(device)
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = torch.mm(features, self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        return localized_features

class Wavelet2NeuralNetwork(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout, ncount):
        super(Wavelet2NeuralNetwork, self).__init__()
        self.dropout = dropout
        self.convolution_1 = SparseGraphWaveletLayer(feat_dim,
                                                     hidden_dim,
                                                     ncount
                                                     )

        self.convolution_2 = SparseGraphWaveletLayer(hidden_dim,
                                                    output_dim,
                                                    ncount
                                                    )
    def forward(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values, device):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        deep_features_1 = self.convolution_1(phi_indices,
                                             phi_values,
                                             phi_inverse_indices,
                                             phi_inverse_values,
                                             feature_indices,
                                             feature_values,
                                             self.dropout,
                                             device)

        deep_features_2 = self.convolution_2(phi_indices,
                                             phi_values,
                                             phi_inverse_indices,
                                             phi_inverse_values,
                                             deep_features_1,
                                             device)

        predictions = torch.nn.functional.log_softmax(deep_features_2, dim=1)
        return predictions