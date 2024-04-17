import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import os
import networkx as nx
import pygsp
import tqdm
import concurrent.futures
from scipy import sparse
from torch_sparse import spspmm, spmm
from sklearn.preprocessing import normalize
from models.utils import scipy_sparse_mat_to_torch_sparse_tensor
from multiprocessing import Pool

class BaseSGModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(BaseSGModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim

        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.processed_feat_list = self.pre_graph_op.propagate(
                adj, feature)
            if self.pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.processed_feature = self.pre_msg_op.aggregate(
                    self.processed_feat_list)

        else: 
            if self.naive_graph_op is not None:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
            self.pre_msg_learnable = False
            self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        if self.base_model.adj != None:
            self.base_model.adj = self.base_model.adj.to(device)
            processed_feature = self.processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori

        else:
            if idx is None and self.processed_feature is not None: idx = torch.arange(self.processed_feature.shape[0])
            if self.pre_msg_learnable is False:
                processed_feature = self.processed_feature[idx].to(device)
            else:
                transferred_feat_list = [feat[idx].to(
                    device) for feat in self.processed_feat_list]
                processed_feature = self.pre_msg_op.aggregate(
                    transferred_feat_list)
            
        output = self.base_model(processed_feature)
        return output[idx] if self.base_model.query_edges is None and self.base_model.adj != None else output

class FeatureAugumentModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(FeatureAugumentModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim

        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None

        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.processed_feat_list = self.pre_graph_op.propagate(
                adj, feature)
            if self.pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.processed_feature = self.pre_msg_op.aggregate(
                    self.processed_feat_list)

        else: 
            if self.naive_graph_op is not None:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
            self.pre_msg_learnable = False
            self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        processed_feature = None
        if self.base_model.adj != None:
            self.base_model.adj = self.base_model.adj.to(device)
            processed_feature = self.processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori

        else:
            if idx is None and self.processed_feature is not None: idx = torch.arange(self.processed_feature.shape[0])
            if self.pre_msg_learnable is False:
                processed_feature = self.processed_feature[idx].to(device)
            else:
                transferred_feat_list = [feat[idx].to(
                    device) for feat in self.processed_feat_list]
                processed_feature = self.pre_msg_op.aggregate(
                    transferred_feat_list)
            
        output = self.base_model(processed_feature)
        return output[idx] if self.base_model.query_edges is None and self.base_model.adj != None else output


class SpectralModel(nn.Module):
    def __init__(self, scale, approximation_order, tolerance):
        super(SpectralModel, self).__init__()
        self.scales = [-scale, scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []
        self.post_graph_op = None

    def preprocess(self, adj, feature):
        self.G = nx.Graph(adj)
        self.index = self.G.nodes()
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.G))
        self.pygsp_graph.estimate_lmax()
        print("\nWavelet calculation and sparsification started.\n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph,
                                                  tau=[scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter,
                                                                              m=self.approximation_order)
            sparsified_wavelets = self.calculate_wavelet()          
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()
        self.ncount = self.phi_matrices[0].shape[0]
        idx = feature.nonzero()
        feature = torch.FloatTensor(feature)
        self.feature_indices = torch.LongTensor(np.vstack(idx))
        self.feature_values = torch.FloatTensor(feature[idx]).view(-1)
        # self.feature_indices = torch.LongTensor([self.feature.row, self.feature.col])
        # self.feature_values = torch.FloatTensor(self.feature.data).view(-1)
        self.phi_indices = torch.LongTensor(self.phi_matrices[0].nonzero())
        self.phi_values = torch.FloatTensor(self.phi_matrices[0][self.phi_matrices[0].nonzero()])
        self.phi_values = self.phi_values.view(-1)
        self.phi_inverse_indices = torch.LongTensor(self.phi_matrices[1].nonzero())
        self.phi_inverse_values = torch.FloatTensor(self.phi_matrices[1][self.phi_matrices[1].nonzero()])
        self.phi_inverse_values = self.phi_inverse_values.view(-1)
        phi_product_indices, phi_product_values = spspmm(self.phi_indices,
                                                         self.phi_values,
                                                         self.phi_inverse_indices,
                                                         self.phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)
        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  feature)
        localized_features = torch.nn.functional.relu(localized_features)
        self.processed_feature = torch.concat((torch.FloatTensor(feature),localized_features),dim = 1)

    # def calculate_wavelet(self):
    #     impulse = np.eye(self.G.number_of_nodes(), dtype=int)
    #     wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
    #                                                                  self.chebyshev,
    #                                                                  impulse)
    #     wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
    #     ind_1, ind_2 = wavelet_coefficients.nonzero()
    #     n_count = self.G.number_of_nodes()
    #     remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
    #                                         shape=(n_count, n_count),
    #                                         dtype=np.float32)
    #     return remaining_waves
    
    def calculate_wavelet(self):
        batch_size = 1000
        n_count = self.G.number_of_nodes()
        wavelet_coefficients = []
        for i in range(0, n_count - batch_size + 1, batch_size):
            impulse = np.zeros((n_count, batch_size))
            impulse[i:i+batch_size, :] = np.eye(batch_size, dtype=int)
            sub_coeffs = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
                                                            self.chebyshev,
                                                            impulse)
            sub_coeffs[sub_coeffs < self.tolerance] = 0
            ind_1, ind_2 = sub_coeffs.nonzero()
            sub_waves = sparse.csr_matrix((sub_coeffs[ind_1, ind_2], (ind_1, ind_2)),
                                        shape=(n_count, batch_size),
                                        dtype=np.float32)
            wavelet_coefficients.append(sub_waves)
        if n_count % batch_size != 0:
            last_batch_size = n_count % batch_size
            impulse = np.zeros((n_count, last_batch_size))
            impulse[-last_batch_size:, :] = np.eye(last_batch_size, dtype=int)
            sub_coeffs = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
                                                            self.chebyshev,
                                                            impulse)
            sub_coeffs[sub_coeffs < self.tolerance] = 0
            ind_1, ind_2 = sub_coeffs.nonzero()
            sub_waves = sparse.csr_matrix((sub_coeffs[ind_1, ind_2], (ind_1, ind_2)),
                                        shape=(n_count, last_batch_size),
                                        dtype=np.float32)
            wavelet_coefficients.append(sub_waves)
        return sparse.hstack(wavelet_coefficients)

    # def calculate_wavelet(self):
    #     ind_1, ind_2 = [], []
    #     data = []
    #     for node in range(self.G.number_of_nodes()):
    #         impulse = np.zeros((self.G.number_of_nodes()))
    #         impulse[node] = 1
    #         wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
    #                                                                         self.chebyshev,
    #                                                                         impulse)
    #         wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
    #         ind_1.extend([node]*len(wavelet_coefficients.nonzero()[0]))
    #         ind_2.extend(wavelet_coefficients.nonzero()[0])
    #         data.extend(wavelet_coefficients[wavelet_coefficients.nonzero()])
    #     n_count = self.G.number_of_nodes()
    #     remaining_waves = sparse.csr_matrix((data, (ind_1, ind_2)),
    #                                         shape=(n_count, n_count),
    #                                         dtype=np.float32)
        
    #     return remaining_waves

    def normalize_matrices(self):
        print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        wavelet_density = len(self.phi_matrices[0].nonzero()[0])/(self.G.number_of_nodes()**2)
        wavelet_density = str(round(100*wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0])/(self.G.number_of_nodes()**2)
        inverse_wavelet_density = str(round(100*inverse_wavelet_density, 2))
        print("Density of wavelets: "+wavelet_density+"%.")
        print("Density of inverse wavelets: "+inverse_wavelet_density+"%.\n")

    def postprocess(self, adj, output):
        return output

    # a wrapper of the forward function
    def model_forward(self, idx, device, ori=None):
        # self.processed_feature = self.processed_feature.to(device)
        return self.forward(idx, device)

    def forward(self, idx, device):  
        #feature = self.processed_feature[idx].to(device)    
        output = self.base_model(self.phi_indices,self.phi_values,self.phi_inverse_indices,self.phi_inverse_values,self.feature_indices,self.feature_values,device)
        return output[idx]

# class ComBaseSGModel(nn.Module):
#     def __init__(self, prop_steps, feat_dim, output_dim):
#         super(ComBaseSGModel, self).__init__()
#         self.prop_steps = prop_steps
#         self.feat_dim = feat_dim
#         self.output_dim = output_dim

#         self.naive_graph_op = None
#         self.pre_graph_op, self.pre_msg_op = None, None
#         self.post_graph_op, self.post_msg_op = None, None
#         self.base_model = None

#         self.real_processed_feat_list = None
#         self.imag_processed_feat_list = None
#         self.real_processed_feature = None
#         self.imag_processed_feature = None
#         self.pre_msg_learnable = False

#     def preprocess(self, adj, feature):
#         if self.pre_graph_op is not None:
#             self.real_processed_feat_list, self.imag_processed_feat_list = \
#             self.pre_graph_op.propagate(adj, feature)
#             if self.pre_msg_op.aggr_type in [
#                 "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                 self.pre_msg_learnable = True
#             else:
#                 self.pre_msg_learnable = False
#                 self.real_processed_feature, self.imag_processed_feature = self.pre_msg_op.aggregate(
#                     self.real_processed_feat_list, self.imag_processed_feat_list)

#         else: 
#             if self.naive_graph_op is not None:
#                 self.base_model.real_adj, self.base_model.imag_adj = self.naive_graph_op.construct_adj(adj)
#                 if not isinstance(self.base_model.real_adj, sp.csr_matrix) or not isinstance(self.base_model.imag_adj, sp.csr_matrix):
#                     raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
#                 elif self.base_model.real_adj.shape[1] != feature.shape[0] or self.base_model.imag_adj.shape[1] != feature.shape[0]:
#                     raise ValueError("Dimension mismatch detected for the real/imag adjacency and the feature matrix!")
#                 self.base_model.real_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.real_adj)
#                 self.base_model.imag_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.imag_adj)
#             self.pre_msg_learnable = False
#             self.real_processed_feature = torch.FloatTensor(feature)
#             self.imag_processed_feature = torch.FloatTensor(feature)

#     def postprocess(self, adj, output):
#         if self.post_graph_op is not None:
#             if self.post_msg_op.aggr_type in [
#                 "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                 raise ValueError(
#                     "Learnable weighted message operator is not supported in the post-processing phase!")
#             output = F.softmax(output, dim=1)
#             output = output.detach().numpy()
#             output = self.post_graph_op.propagate(adj, output)
#             output = self.post_msg_op.aggregate(output)

#         return output

#     # a wrapper of the forward function
#     def model_forward(self, idx, device, ori=None):
#         return self.forward(idx, device, ori)

#     def forward(self, idx, device, ori):
#         real_processed_feature = None
#         imag_processed_feature = None
#         if self.base_model.real_adj != None or self.base_model.imag_adj != None:
#             self.base_model.real_adj = self.base_model.real_adj.to(device)
#             self.base_model.imag_adj = self.base_model.imag_adj.to(device)
#             real_processed_feature = self.real_processed_feature.to(device)
#             imag_processed_feature = self.imag_processed_feature.to(device)
#             if ori is not None: self.base_model.query_edges = ori

#         else:
#             if idx is None and self.real_processed_feature is not None: idx = torch.arange(self.real_processed_feature.shape[0])
#             if self.pre_msg_learnable is False:
#                 real_processed_feature = self.real_processed_feature[idx].to(device)
#                 imag_processed_feature = self.imag_processed_feature[idx].to(device)
#             else:
#                 real_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feature]
#                 imag_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feature]
#                 real_processed_feature, imag_processed_feature = self.pre_msg_op.aggregate(
#                     real_transferred_feat_list, imag_transferred_feat_list)
                
#         output = self.base_model(real_processed_feature, imag_processed_feature)
#         return output[idx] if (self.base_model.query_edges is None and (self.base_model.real_adj != None or self.base_model.imag_adj != None)) else output


# class ComBaseMultiPropSGModel(nn.Module):
#     def __init__(self, prop_steps, feat_dim, output_dim):
#         super(ComBaseMultiPropSGModel, self).__init__()
#         self.prop_steps = prop_steps
#         self.feat_dim = feat_dim
#         self.output_dim = output_dim

#         self.pre_graph_op_list, self.pre_msg_op_list = [], []
#         self.post_graph_op, self.post_msg_op = None, None
#         self.pre_multi_msg_op = None
#         self.base_model = None

#         self.real_processed_feat_list = None
#         self.real_processed_feat_list_list = []
#         self.imag_processed_feat_list = None
#         self.imag_processed_feat_list_list = []

#         self.real_processed_feature = None
#         self.real_processed_feature_list = []
#         self.imag_processed_feature = None
#         self.imag_processed_feature_list = []
#         self.pre_msg_learnable_list = []
#         self.pre_multi_msg_learnable = None

#     def preprocess(self, adj, feature):
#         if len(self.pre_graph_op_list) != 0 and (len(self.pre_graph_op_list) == len(self.pre_msg_op_list)):
#             for i in range(len(self.pre_graph_op_list)):
#                 if self.pre_msg_op_list[i].aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                     self.pre_msg_learnable_list.append(True)
#                 else:
#                     self.pre_msg_learnable_list.append(False)

#             if not self.pre_msg_learnable_list.count(self.pre_msg_learnable_list[0]) == len(self.pre_msg_learnable_list):
#                 raise ValueError("In the current version only multi-operator same message aggregation patterns (learned, unlearnable) are supported!")
            
#             if self.pre_msg_op_list[0].aggr_type == "identity" and all(_ == "identity" for _ in self.pre_msg_learnable_list) is False:
#                 raise ValueError("In the current version the mix of identity mapping operators and other operators is not supported!")

#             for i in range(len(self.pre_graph_op_list)):
#                 self.real_processed_feat_list, self.imag_processed_feat_list = self.pre_graph_op_list[i].propagate(adj, feature)
#                 self.real_processed_feat_list_list.append(self.real_processed_feat_list)
#                 self.imag_processed_feat_list_list.append(self.imag_processed_feat_list)

#                 if self.pre_msg_learnable_list[i] is False:
#                     self.real_processed_feature, self.imag_processed_feature = self.pre_msg_op_list[i].aggregate(self.real_processed_feat_list_list[-1], self.imag_processed_feat_list_list[-1])
#                     if self.pre_msg_op_list[i].aggr_type in ["identity"]:
#                         self.real_processed_feature_list.extend(self.real_processed_feature)
#                         self.imag_processed_feature_list.extend(self.imag_processed_feature)
#                     else:
#                         self.real_processed_feature_list.append(self.real_processed_feature)
#                         self.imag_processed_feature_list.append(self.imag_processed_feature)

#             if self.pre_multi_msg_op is not None: 
#                 if self.pre_multi_msg_op.aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                     self.pre_multi_msg_learnable = True
#                 else:
#                     self.pre_multi_msg_learnable = False
#                     if self.pre_msg_op_list[0].aggr_type == "identity":
#                         self.real_processed_feature, self.imag_processed_feature =  self.pre_multi_msg_op.aggregate(self.real_processed_feature_list, self.imag_processed_feature_list)

#             else:
#                 self.pre_multi_msg_learnable = False
#         else:
#             raise ValueError("MultiProp must define One-to-One propagation operator!")
        
#     def postprocess(self, adj, output):
#         if self.post_graph_op is not None:
#             if self.post_msg_op.aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                 raise ValueError("Learnable weighted message operator is not supported in the post-processing phase!")
#             output = F.softmax(output, dim=1)
#             output = output.detach().numpy()
#             output = self.post_graph_op.propagate(adj, output)
#             output = self.post_msg_op.aggregate(output)

#         return output

#     # a wrapper of the forward function
#     def model_forward(self, idx, device, ori=None):
#         return self.forward(idx, device, ori)

#     def forward(self, idx, device, ori):
#         real_processed_feature = None
#         imag_processed_feature = None
            
#         # f f
#         if idx is None and self.real_processed_feature is not None: idx = torch.arange(self.real_processed_feature.shape[0])
#         if all(_ is True for _ in self.pre_msg_learnable_list) is False and self.pre_multi_msg_learnable is False:
#             real_processed_feature = self.real_processed_feature[idx].to(device)
#             imag_processed_feature = self.imag_processed_feature[idx].to(device)
#         # f t
#         elif all(_ is True for _ in self.pre_msg_learnable_list) is False and self.pre_multi_msg_learnable is True:
#             real_multi_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feature_list]
#             imag_multi_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feature_list]
#             real_processed_feature, imag_processed_feature = self.pre_multi_msg_op.aggregate(real_multi_transferred_feat_list, imag_multi_transferred_feat_list)
#         # t f / t t
#         else:
#             self.real_processed_feature_list = []
#             self.imag_processed_feature_list = []
#             for i in range(len(self.real_processed_feat_list_list)):
#                 self.pre_msg_op_list[i] =  self.pre_msg_op_list[i].to(device)
#                 real_transferred_feat_list = [feat[idx].to(device) for feat in self.real_processed_feat_list_list[i]]
#                 imag_transferred_feat_list = [feat[idx].to(device) for feat in self.imag_processed_feat_list_list[i]]
#                 real_processed_feature, imag_processed_feature = self.pre_msg_op_list[i].aggregate(real_transferred_feat_list, imag_transferred_feat_list)
#                 self.real_processed_feature_list.append(real_processed_feature)
#                 self.imag_processed_feature_list.append(imag_processed_feature)
#             real_processed_feature, imag_processed_feature = self.pre_multi_msg_op.aggregate(self.real_processed_feature_list, self.imag_processed_feature_list)
        
#         output = self.base_model(real_processed_feature, imag_processed_feature)
        
#         return output
    

# class TwoOrderBaseSGModel(nn.Module):
#     def __init__(self, prop_steps, feat_dim, output_dim):
#         super(TwoOrderBaseSGModel, self).__init__()
#         self.prop_steps = prop_steps
#         self.feat_dim = feat_dim
#         self.output_dim = output_dim

#         self.naive_graph_op = None
#         self.pre_graph_op, self.pre_msg_op = None, None
#         self.post_graph_op, self.post_msg_op = None, None
#         self.base_model = None

#         self.one_processed_feat_list = None
#         self.two_processed_feat_list = None
#         self.one_processed_feature = None
#         self.two_processed_feature = None

#         self.pre_msg_learnable = False

#     def preprocess(self, adj, feature):
#         if self.naive_graph_op is not None:
#             self.base_model.one_adj, self.base_model.two_adj = self.naive_graph_op.construct_adj(adj)
#             if not isinstance(self.base_model.one_adj, sp.csr_matrix) or not isinstance(self.base_model.two_adj, sp.csr_matrix):
#                 raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
#             elif self.base_model.one_adj.shape[1] != feature.shape[0] or self.base_model.two_adj.shape[1] != feature.shape[0]:
#                 raise ValueError("Dimension mismatch detected for the un/in/out adjacency and the feature matrix!")
#             self.base_model.one_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.one_adj)
#             self.base_model.two_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.two_adj)
#         else:
#             raise ValueError("TwoOrderBaseSGModel must predefine the graph structure operator!")
        
#         self.pre_msg_learnable = False
#         self.original_feature = torch.FloatTensor(feature)
#         self.one_processed_feature = torch.FloatTensor(feature)
#         self.two_processed_feature = torch.FloatTensor(feature)

#     def postprocess(self, adj, output):
#         if self.post_graph_op is not None:
#             if self.post_msg_op.aggr_type in [
#                 "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                 raise ValueError(
#                     "Learnable weighted message operator is not supported in the post-processing phase!")
#             output = F.softmax(output, dim=1)
#             output = output.detach().numpy()
#             output = self.post_graph_op.propagate(adj, output)
#             output = self.post_msg_op.aggregate(output)

#         return output

#     # a wrapper of the forward function
#     def model_forward(self, idx, device, ori=None):
#         return self.forward(idx, device, ori)

#     def forward(self, idx, device, ori):
#         one_processed_feature = None
#         two_processed_feature = None
#         self.base_model.one_adj = self.base_model.one_adj.to(device)
#         self.base_model.two_adj = self.base_model.two_adj.to(device)
#         one_processed_feature = self.one_processed_feature.to(device)
#         two_processed_feature = self.two_processed_feature.to(device)
#         original_feature = self.original_feature.to(device)
#         if ori is not None: self.base_model.query_edges = ori
#         output = self.base_model(original_feature, one_processed_feature, two_processed_feature)
#         return output[idx] if self.base_model.query_edges is None else output
    

# class TwoDirBaseSGModel(nn.Module):
#     def __init__(self, prop_steps, feat_dim, output_dim):
#         super(TwoDirBaseSGModel, self).__init__()
#         self.prop_steps = prop_steps
#         self.feat_dim = feat_dim
#         self.output_dim = output_dim

#         self.naive_graph_op = None
#         self.pre_graph_op, self.pre_msg_op = None, None
#         self.post_graph_op, self.post_msg_op = None, None
#         self.base_model = None

#         self.un_processed_feat_list = None
#         self.in_processed_feat_list = None
#         self.out_processed_feat_list = None
#         self.un_processed_feature = None
#         self.in_processed_feature = None
#         self.out_processed_feature = None
#         self.pre_msg_learnable = False

#     def preprocess(self, adj, feature): 
#         if self.naive_graph_op is not None:
#             self.base_model.un_adj, self.base_model.in_adj, self.base_model.out_adj = self.naive_graph_op.construct_adj(adj)
#             if not isinstance(self.base_model.un_adj, sp.csr_matrix) or not isinstance(self.base_model.in_adj, sp.csr_matrix) or not isinstance(self.base_model.out_adj, sp.csr_matrix):
#                 raise TypeError("The real/imag adjacency matrix must be a scipy csr sparse matrix!")
#             elif self.base_model.un_adj.shape[1] != feature.shape[0] or self.base_model.in_adj.shape[1] != feature.shape[0] or self.base_model.out_adj.shape[1] != feature.shape[0]:
#                 raise ValueError("Dimension mismatch detected for the un/in/out adjacency and the feature matrix!")
#             self.base_model.un_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.un_adj)
#             self.base_model.in_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.in_adj)
#             self.base_model.out_adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.out_adj)
#         else:
#             raise ValueError("TwoDirBaseSGModel must predefine the graph structure operator!")
        
#         self.pre_msg_learnable = False
#         self.un_processed_feature = torch.FloatTensor(feature)
#         self.in_processed_feature = torch.FloatTensor(feature)
#         self.out_processed_feature = torch.FloatTensor(feature)

#     def postprocess(self, adj, output):
#         if self.post_graph_op is not None:
#             if self.post_msg_op.aggr_type in [
#                 "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
#                 raise ValueError(
#                     "Learnable weighted message operator is not supported in the post-processing phase!")
#             output = F.softmax(output, dim=1)
#             output = output.detach().numpy()
#             output = self.post_graph_op.propagate(adj, output)
#             output = self.post_msg_op.aggregate(output)

#         return output

#     # a wrapper of the forward function
#     def model_forward(self, idx, device, ori=None):
#         return self.forward(idx, device, ori)

#     def forward(self, idx, device, ori):
#         un_processed_feature = None
#         in_processed_feature = None
#         out_processed_feature = None
#         self.base_model.un_adj = self.base_model.un_adj.to(device)
#         self.base_model.in_adj = self.base_model.in_adj.to(device)
#         self.base_model.out_adj = self.base_model.out_adj.to(device)
#         un_processed_feature = self.un_processed_feature.to(device)
#         in_processed_feature = self.in_processed_feature.to(device)
#         out_processed_feature = self.out_processed_feature.to(device)
#         if ori is not None: self.base_model.query_edges = ori
#         output = self.base_model(un_processed_feature, in_processed_feature, out_processed_feature)
#         return output[idx] if self.base_model.query_edges is None else output