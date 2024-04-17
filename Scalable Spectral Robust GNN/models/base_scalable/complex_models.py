# import torch
# import torch.nn as nn

# from models.base_scalable.complex_relu import ComReLU

# class ComOneDimConvolution(nn.Module):
#     def __init__(self, num_subgraphs, prop_steps, feat_dim):
#         super(ComOneDimConvolution, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.hop_num = prop_steps

#         self.real_learnable_weight = nn.ParameterList()
#         for _ in range(prop_steps):
#             self.real_learnable_weight.append(nn.Parameter(
#                 torch.FloatTensor(feat_dim, num_subgraphs)))
            
#         self.imag_learnable_weight = nn.ParameterList()
#         for _ in range(prop_steps):
#             self.imag_learnable_weight.append(nn.Parameter(
#                 torch.FloatTensor(feat_dim, num_subgraphs)))
#         self.reset_parameters()

#     def reset_parameters(self):
#         for weight in self.real_learnable_weight:
#             nn.init.xavier_uniform_(weight)

#         for weight in self.imag_learnable_weight:
#             nn.init.xavier_uniform_(weight)

#     # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
#     def forward(self, real_feat_list_list, imag_feat_list_list):
#         real_aggregated_feat_list = []
#         for i in range(self.hop_num):
#             adopted_feat = torch.stack(real_feat_list_list[i], dim=2)
#             intermediate_feat = (
#                     adopted_feat * (self.real_learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

#             real_aggregated_feat_list.append(intermediate_feat)

#         imag_aggregated_feat_list = []
#         for i in range(self.hop_num):
#             adopted_feat = torch.stack(imag_feat_list_list[i], dim=2)
#             intermediate_feat = (
#                     adopted_feat * (self.imag_learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

#             imag_aggregated_feat_list.append(intermediate_feat)
#         return real_aggregated_feat_list, imag_aggregated_feat_list


# class ComOneDimConvolutionWeightSharedAcrossFeatures(nn.Module):
#     def __init__(self, num_subgraphs, prop_steps):
#         super(ComOneDimConvolutionWeightSharedAcrossFeatures, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.hop_num = prop_steps

#         self.real_learnable_weight = nn.ParameterList()
#         for _ in range(prop_steps):
#             # To help xvarient_uniform_ calculate fan in and fan out, "1" should be kept here.
#             self.real_learnable_weight.append(nn.Parameter(
#                 torch.FloatTensor(1, num_subgraphs)))

#         self.imag_learnable_weight = nn.ParameterList()
#         for _ in range(prop_steps):
#             # To help xvarient_uniform_ calculate fan in and fan out, "1" should be kept here.
#             self.imag_learnable_weight.append(nn.Parameter(
#                 torch.FloatTensor(1, num_subgraphs)))
            
#         self.reset_parameters()

#     def reset_parameters(self):
#         for weight in self.real_learnable_weight:
#             nn.init.xavier_uniform_(weight)

#         for weight in self.imag_learnable_weight:
#             nn.init.xavier_uniform_(weight)

#     # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
#     def forward(self, real_feat_list_list, imag_feat_list_list):
#         real_aggregated_feat_list = []
#         for i in range(self.hop_num):
#             adopted_feat = torch.stack(real_feat_list_list[i], dim=2)
#             intermediate_feat = (
#                     adopted_feat * (self.real_learnable_weight[i])).mean(dim=2)

#             real_aggregated_feat_list.append(intermediate_feat)

#         imag_aggregated_feat_list = []
#         for i in range(self.hop_num):
#             adopted_feat = torch.stack(imag_feat_list_list[i], dim=2)
#             intermediate_feat = (
#                     adopted_feat * (self.imag_learnable_weight[i])).mean(dim=2)

#             imag_aggregated_feat_list.append(intermediate_feat)
#         return real_aggregated_feat_list, imag_aggregated_feat_list


# class ComFastOneDimConvolution(nn.Module):
#     def __init__(self, num_subgraphs, prop_steps):
#         super(ComFastOneDimConvolution, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.num_subgraphs = num_subgraphs
#         self.prop_steps = prop_steps

#         # How to initialize the weight is extremely important.
#         # Pure xavier will lead to extremely unstable accuracy.
#         # Initialized with ones will not perform as good as this one.        
#         self.real_learnable_weight = nn.Parameter(
#             torch.ones(num_subgraphs * prop_steps, 1))

#         self.imag_learnable_weight = nn.Parameter(
#             torch.ones(num_subgraphs * prop_steps, 1))
        
#     # feat_list_list: 3-d tensor (num_node, feat_dim, num_subgraphs * prop_steps)
#     def forward(self, real_feat_list_list, imag_feat_list_list):
#         return (real_feat_list_list @ self.real_learnable_weight).squeeze(dim=2), (imag_feat_list_list @ self.imag_learnable_weight).squeeze(dim=2)

#     def subgraph_weight(self):
#         return  self.real_learnable_weight.view(self.num_subgraphs, self.prop_steps).sum(dim=1),\
#                 self.imag_learnable_weight.view(self.num_subgraphs, self.prop_steps).sum(dim=1)


# class ComIdenticalMapping(nn.Module):
#     def __init__(self) -> None:
#         super(ComIdenticalMapping, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None

#     def forward(self, real_feature, imag_feature):
#         return real_feature, imag_feature


# class ComLogisticRegression(nn.Module):
#     def __init__(self, feat_dim, output_dim, dropout):
#         super(ComLogisticRegression, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.query_edges = None

#         self.dropout = nn.Dropout(dropout)
#         self.real_fc = nn.Linear(feat_dim, output_dim)
#         self.imag_fc = nn.Linear(feat_dim, output_dim)
#         # self.one_dim_conv = nn.Conv1d(2*feat_dim, output_dim, kernel_size=1)
#         self.one_dim_conv = nn.Linear(2*output_dim, output_dim)
#         self.real_imag_linear = nn.Linear(output_dim*4, output_dim)

#     def forward(self, real_feature, imag_feature):
#         real_feature = self.real_fc(real_feature)
#         imag_feature = self.imag_fc(imag_feature)
#         real_x = real_feature
#         imag_x = imag_feature
#         if self.query_edges == None:
#             x = torch.cat((real_x, imag_x), dim=-1)
#             x = self.dropout(x)
#             # x = x.unsqueeze(0)
#             # x = x.permute((0, 2, 1))
#             x = self.one_dim_conv(x)
#             # return torch.transpose(x[0], 0, 1)
#             return x
#         else:
#             x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]],
#                         imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.real_imag_linear(x)
#             return x


# class ComMultiLayerPerceptron(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout, bn=False):
#         super(ComMultiLayerPerceptron, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.query_edges = None

#         if num_layers < 2:
#             raise ValueError("MLP must have at least two layers!")
#         self.num_layers = num_layers

#         self.real_fcs = nn.ModuleList()
#         self.real_fcs.append(nn.Linear(feat_dim, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.real_fcs.append(nn.Linear(hidden_dim, hidden_dim))

#         self.imag_fcs = nn.ModuleList()
#         self.imag_fcs.append(nn.Linear(feat_dim, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.imag_fcs.append(nn.Linear(hidden_dim, hidden_dim))

#         self.bn = bn
#         if self.bn is True:
#             self.bns = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.bns.append(nn.BatchNorm1d(hidden_dim))

#         self.dropout = nn.Dropout(dropout)
#         self.comrelu = ComReLU()
#         # self.one_dim_conv = nn.Conv1d(2*hidden_dim, output_dim, kernel_size=1)
#         self.one_dim_conv = nn.Linear(2*hidden_dim, output_dim)
#         self.real_imag_linear = nn.Linear(hidden_dim*4, output_dim)
#         self.reset_parameters()

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain("relu")
#         for fc in self.real_fcs:
#             nn.init.xavier_uniform_(fc.weight, gain=gain)
#             nn.init.zeros_(fc.bias)

#         for fc in self.imag_fcs:
#             nn.init.xavier_uniform_(fc.weight, gain=gain)
#             nn.init.zeros_(fc.bias)

#     def forward(self, real_feature, imag_feature):
#         for i in range(self.num_layers - 1):
#             real_feature = self.real_fcs[i](real_feature)
#             imag_feature = self.imag_fcs[i](imag_feature)
#             if self.bn is True:
#                 real_feature = self.bns[i](real_feature)
#                 imag_feature = self.bns[i](imag_feature)
#             real_feature, imag_feature = self.comrelu(real_feature, imag_feature)
#             real_feature = self.dropout(real_feature)
#             imag_feature = self.dropout(imag_feature)

#         real_x = real_feature
#         imag_x = imag_feature
#         if self.query_edges == None:
#             x = torch.cat((real_x, imag_x), dim=-1)
#             x = self.dropout(x)
#             # x = x.unsqueeze(0)
#             # x = x.permute((0, 2, 1))
#             x = self.one_dim_conv(x)
#             # return torch.transpose(x[0], 0, 1)
#             return x
#         else:
#             x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]],
#                         imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.real_imag_linear(x)
#             return x


# class ComResMultiLayerPerceptron(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False):
#         super(ComResMultiLayerPerceptron, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.query_edges = None

#         if num_layers < 2:
#             raise ValueError("ResMLP must have at least two layers!")
#         self.num_layers = num_layers

#         self.real_fcs = nn.ModuleList()
#         self.real_fcs.append(nn.Linear(feat_dim, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.real_fcs.append(nn.Linear(hidden_dim, hidden_dim))

#         self.imag_fcs = nn.ModuleList()
#         self.imag_fcs.append(nn.Linear(feat_dim, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.imag_fcs.append(nn.Linear(hidden_dim, hidden_dim))

#         self.bn = bn
#         if self.bn is True:
#             self.bns = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.bns.append(nn.BatchNorm1d(hidden_dim))

#         self.dropout = nn.Dropout(dropout)
#         self.comrelu = ComReLU()
#         # self.one_dim_conv = nn.Conv1d(2*hidden_dim, output_dim, kernel_size=1)
#         self.one_dim_conv = nn.Linear(2*hidden_dim, output_dim)
#         self.real_imag_linear = nn.Linear(hidden_dim*4, output_dim)

#     def forward(self, real_feature, imag_feature):
#         real_feature = self.dropout(real_feature)
#         imag_feature = self.dropout(imag_feature)

#         real_feature = self.real_fcs[0](real_feature)
#         imag_feature = self.imag_fcs[0](imag_feature)

#         if self.bn is True:
#             real_feature = self.bns[0](real_feature)
#             imag_feature = self.bns[0](imag_feature)
#         real_feature, imag_feature = self.comrelu(real_feature, imag_feature)
#         real_residual = real_feature
#         imag_residual = imag_feature

#         for i in range(1, self.num_layers - 1):
#             real_feature = self.dropout(real_feature)
#             imag_feature = self.dropout(imag_feature)
#             real_feature = self.real_fcs[i](real_feature)
#             imag_feature = self.imag_fcs[i](imag_feature)
#             if self.bn is True:
#                 real_feature = self.bns[i](real_feature)
#                 imag_feature = self.bns[i](imag_feature)
#             real_feature_, imag_feature_ = self.comrelu(real_feature, imag_feature)
#             real_feature = real_feature_ + real_residual
#             imag_feature = imag_feature_ + imag_residual
#             real_residual = real_feature_
#             imag_residual = imag_feature_

#         real_feature = self.dropout(real_feature)
#         imag_feature = self.dropout(imag_feature)

#         real_x = real_feature
#         imag_x = imag_feature
#         if  self.query_edges == None:
#             x = torch.cat((real_x, imag_x), dim=-1)
#             x = self.dropout(x)
#             # x = x.unsqueeze(0)
#             # x = x.permute((0, 2, 1))
#             x = self.one_dim_conv(x)
#             # return torch.transpose(x[0], 0, 1)
#             return x
#         else:
#             x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]],
#                         imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.real_imag_linear(x)
#             return x


# class Com2LayerGraphConvolution(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
#         super(Com2LayerGraphConvolution, self).__init__()
#         self.real_adj = None
#         self.imag_adj = None
#         self.query_edges = None
        
#         self.comrelu = ComReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.real_imag_fc1 = nn.Linear(feat_dim, hidden_dim)
#         self.real_imag_prop_fc1 = nn.Linear(feat_dim, hidden_dim)
#         self.real_imag_fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.real_imag_prop_fc2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.one_dim_conv = nn.Conv1d(2*hidden_dim, output_dim, kernel_size=1)
#         self.one_dim_conv = nn.Linear(2*hidden_dim, output_dim)
#         self.real_imag_linear = nn.Linear(hidden_dim*4, output_dim)

#     def forward(self, real_feature, imag_feature):
#         real_real_x, imag_imag_x = real_feature, imag_feature
#         imag_real_x, real_imag_x = real_feature, imag_feature
#         real_real_x, imag_imag_x = self.real_imag_fc1(real_real_x), self.real_imag_fc1(imag_imag_x)
#         imag_real_x, real_imag_x = self.real_imag_fc1(imag_real_x), self.real_imag_fc1(real_imag_x)

#         real_real_x_prop, imag_imag_x_prop = torch.mm(self.real_adj, real_feature), torch.mm(self.imag_adj, imag_feature)
#         imag_real_x_prop, real_imag_x_prop = torch.mm(self.real_adj, real_feature), torch.mm(self.imag_adj, imag_feature)

#         real_real_x = real_real_x + self.real_imag_prop_fc1(real_real_x_prop)
#         imag_imag_x = imag_imag_x + self.real_imag_prop_fc1(imag_imag_x_prop)
#         imag_real_x = imag_real_x + self.real_imag_prop_fc1(imag_real_x_prop)
#         real_imag_x = real_imag_x + self.real_imag_prop_fc1(real_imag_x_prop)

#         layer1_out_real = real_real_x - imag_imag_x
#         layer1_out_imag = imag_real_x + real_imag_x
#         layer1_out_real, layer1_out_imag = self.comrelu(layer1_out_real, layer1_out_imag)
#         layer1_out_real, layer1_out_imag = self.dropout(layer1_out_real), self.dropout(layer1_out_imag)

#         real_real_x, imag_imag_x = layer1_out_real, layer1_out_imag
#         imag_real_x, real_imag_x = layer1_out_real, layer1_out_imag
#         real_real_x, imag_imag_x = self.real_imag_fc2(real_real_x), self.real_imag_fc2(imag_imag_x)
#         imag_real_x, real_imag_x = self.real_imag_fc2(imag_real_x), self.real_imag_fc2(real_imag_x)

#         real_real_x_prop, imag_imag_x_prop = torch.mm(self.real_adj, layer1_out_real), torch.mm(self.imag_adj, layer1_out_imag)
#         imag_real_x_prop, real_imag_x_prop = torch.mm(self.real_adj, layer1_out_real), torch.mm(self.imag_adj, layer1_out_imag)

#         real_real_x = real_real_x + self.real_imag_prop_fc2(real_real_x_prop)
#         imag_imag_x = imag_imag_x + self.real_imag_prop_fc2(imag_imag_x_prop)
#         imag_real_x = imag_real_x + self.real_imag_prop_fc2(imag_real_x_prop)
#         real_imag_x = real_imag_x + self.real_imag_prop_fc2(real_imag_x_prop)

#         layer2_out_real = real_real_x - imag_imag_x
#         layer2_out_imag = imag_real_x + real_imag_x
#         real_x, imag_x = self.comrelu(layer2_out_real, layer2_out_imag)

#         if self.query_edges == None:
#             x = torch.cat((real_x, imag_x), dim=-1)
#             x = self.dropout(x)
#             # x = x.unsqueeze(0)
#             # x = x.permute((0, 2, 1))
#             x = self.one_dim_conv(x)
#             # return torch.transpose(x[0], 0, 1)
#             return x
#         else:
#             x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]],
#                         imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.real_imag_linear(x)
#             return x


# class TwoDir2LayerGraphConvolution(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
#         super(TwoDir2LayerGraphConvolution, self).__init__()
#         self.un_adj = None
#         self.in_adj = None
#         self.out_adj = None
#         self.query_edges = None
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.lin1 = nn.Linear(feat_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim*3, hidden_dim)

#         # self.Conv = nn.Conv1d(hidden*3, label_dim, kernel_size=1)
#         self.one_dim_conv = nn.Linear(3*hidden_dim, output_dim)
#         self.linear = nn.Linear(hidden_dim*6, output_dim)

#     def forward(self, un_feature, in_feature, out_feature):
#         x_un = self.lin1(un_feature)
#         x_in = self.lin1(in_feature)
#         x_out = self.lin1(out_feature)
#         x_un = torch.mm(self.un_adj, x_un)
#         x_in = torch.mm(self.in_adj, x_in)
#         x_out = torch.mm(self.out_adj, x_out)

#         x = torch.cat((x_un, x_in, x_out), axis=-1)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.lin2(x)
#         x_un = torch.mm(self.un_adj, x)
#         x_in = torch.mm(self.in_adj, x)
#         x_out = torch.mm(self.out_adj, x)

#         x = torch.cat((x_un, x_in, x_out), axis=-1)
#         x = self.relu(x)
#         if self.query_edges == None:
#             x = self.dropout(x)
#             x = self.one_dim_conv(x)
#             # x = x.unsqueeze(0)
#             # x = x.permute((0, 2, 1))
#             # x = self.Conv(x)
#             # x = x.permute((0, 2, 1)).squeeze()
#             return x
#         else:
#             x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.linear(x)
#             return x


# class FastPprApprox2LayerGraphConvolution(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
#         super(FastPprApprox2LayerGraphConvolution, self).__init__()
#         self.adj = None
#         self.query_edges = None
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.lin1 = nn.Linear(feat_dim, hidden_dim)
#         self.lin2_node = nn.Linear(hidden_dim, output_dim)
#         self.lin2_edge = nn.Linear(hidden_dim, hidden_dim)
#         self.linear = nn.Linear(hidden_dim*2, output_dim)

#     def forward(self, feature):
#         x = self.lin1(feature)
#         x = torch.mm(self.adj, x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         if self.query_edges == None:
#             x = self.lin2_node(x)
#             x = torch.mm(self.adj, x)
#             return x
#         else:
#             x = self.lin2_edge(x)
#             x = torch.mm(self.adj, x)
#             x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)            
#             x = self.dropout(x)
#             x = self.linear(x)
#             return x


# class TwoOrderPprApprox2LayerGraphConvolution(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
#         super(TwoOrderPprApprox2LayerGraphConvolution, self).__init__()
#         self.one_adj = None
#         self.two_adj = None
#         self.query_edges = None
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.lin1 = nn.Linear(feat_dim, hidden_dim)
#         self.lin1_1 = nn.Linear(feat_dim, hidden_dim)
#         self.lin1_2 = nn.Linear(feat_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin2_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin2_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin3_node = nn.Linear(hidden_dim, output_dim)
#         self.lin3_1_node = nn.Linear(hidden_dim, output_dim)
#         self.lin3_2_node = nn.Linear(hidden_dim, output_dim)
#         self.lin3_edge = nn.Linear(hidden_dim, hidden_dim)
#         self.lin3_1_edge = nn.Linear(hidden_dim, hidden_dim)
#         self.lin3_2_edge = nn.Linear(hidden_dim, hidden_dim)
#         self.linear = nn.Linear(hidden_dim*2, output_dim)

#     def forward(self, original_feature, one_processed_feature, two_processed_feature):
#         x_0 = self.lin1(original_feature)
#         x_1 = self.lin1_1(one_processed_feature)
#         x_1 = torch.mm(self.one_adj, x_1) 
#         x_2 = self.lin1_2(two_processed_feature)
#         x_2 = torch.mm(self.two_adj, x_2) 
#         x_0 = self.relu(x_0)
#         x_1 = self.relu(x_1)
#         x_2 = self.relu(x_2)
#         x_0 = self.dropout(x_0)
#         x_1 = self.dropout(x_1)
#         x_2 = self.dropout(x_2)
#         x = x_0 + x_1 + x_2
        
#         x_0 = self.lin2(x)
#         x_1 = self.lin2_1(x)
#         x_1 = torch.mm(self.one_adj, x_1) 
#         x_2 = self.lin2_2(x)
#         x_2 = torch.mm(self.two_adj, x_2) 
#         x_0 = self.relu(x_0)
#         x_1 = self.relu(x_1)
#         x_2 = self.relu(x_2)
#         x_0 = self.dropout(x_0)
#         x_1 = self.dropout(x_1)
#         x_2 = self.dropout(x_2)
#         x = x_0 + x_1 + x_2

#         if self.query_edges == None:
#             x_0 = self.lin3_node(x)
#             x_1 = self.lin3_1_node(x)
#             x_1 = torch.mm(self.one_adj, x_1) 
#             x_2 = self.lin3_2_node(x)
#             x_2 = torch.mm(self.two_adj, x_2) 
#             x_0 = self.relu(x_0)
#             x_1 = self.relu(x_1)
#             x_2 = self.relu(x_2)
#             x_0 = self.dropout(x_0)
#             x_1 = self.dropout(x_1)
#             x_2 = self.dropout(x_2)
#             x = x_0 + x_1 + x_2
#             return x
#         else:
#             x_0 = self.lin3_edge(x)
#             x_1 = self.lin3_1_edge(x)
#             x_1 = torch.mm(self.one_adj, x_1) 
#             x_2 = self.lin3_2_edge(x)
#             x_2 = torch.mm(self.two_adj, x_2) 
#             x_0 = self.relu(x_0)
#             x_1 = self.relu(x_1)
#             x_2 = self.relu(x_2)
#             x_0 = self.dropout(x_0)
#             x_1 = self.dropout(x_1)
#             x_2 = self.dropout(x_2)
#             x = x_0 + x_1 + x_2
#             x = torch.cat((x[self.query_edges[:, 0]], x[self.query_edges[:, 1]]), dim=-1)
#             x = self.dropout(x)
#             x = self.linear(x)
#             return x