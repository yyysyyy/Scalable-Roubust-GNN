import torch
import platform
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from torch import Tensor
from operators.utils import csr_sparse_dense_matmul, cuda_csr_sparse_dense_matmul


class GraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj = self.construct_adj(adj)

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        prop_feat_list = [feature]
        for _ in range(self.prop_steps):
            feat_temp = ada_platform_one_step_propagation(self.adj, prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
        return [torch.FloatTensor(feat) for feat in prop_feat_list]


# Might include training parameters
class MessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(MessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self.combine(feat_list)
    

class TwoOrderPprApproxGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.one_adj = None
        self.two_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.one_adj, self.two_adj = self.construct_adj(adj)
        one_prop_feat_list = []
        two_prop_feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.one_adj.shape[1] != feature.shape[0] or self.two_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        one_prop_feat_list = [feature]
        two_prop_feat_list = [feature]

        for _ in range(self.prop_steps):
            one_feat_temp = ada_platform_one_step_propagation(self.one_adj, one_prop_feat_list[-1])
            two_feat_temp = ada_platform_one_step_propagation(self.two_adj, two_prop_feat_list[-1])
            one_prop_feat_list.append(one_feat_temp)
            two_prop_feat_list.append(two_feat_temp)

        return [torch.FloatTensor(feat) for feat in one_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in two_prop_feat_list]

# Might include training parameters
class TwoOrderPprApproxMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoOrderPprApproxMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, one_feat_list, two_feat_list):
        return NotImplementedError

    def aggregate(self, one_feat_list, two_feat_list):
        if not isinstance(one_feat_list, list) or not isinstance(two_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in one_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The one order feature matrices must be tensors!")
        for feat in two_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The two order feature matrices must be tensors!")
            
        return self.combine(one_feat_list, two_feat_list)


class calculator:
    def __init__(self, value, r_step=0, i_step=0):
        self.value = value
        self.r_step = r_step
        self.i_step = i_step

    def prop_step(self):
        return self.r_step + self.i_step
    
    def reversal(self):
        if self.i_step & 1 == 0 and self.i_step != 0:
            self.value = -self.value
    
    def set_variable(self, value, r=False, i=False):
        self.value = value
        if r:   self.r_step += 1
        elif i: self.i_step += 1


class ComGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.real_adj = None
        self.imag_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.real_adj, self.imag_adj = self.construct_adj(adj)

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.real_adj.shape[1] != feature.shape[0] or self.imag_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        init_real_calculator = calculator(feature)
        init_imag_calculator = calculator(feature)

        real_prop_feat_list = [init_real_calculator.value]
        imag_prop_feat_list = [init_imag_calculator.value]
        tmp_prop_feat_calculator_in_list = []
        tmp_prop_feat_calculator_out_list = []

        for steps in range(self.prop_steps):
            if steps == 0:
                real_feat_temp_value = ada_platform_one_step_propagation(self.real_adj, real_prop_feat_list[-1])
                init_real_calculator.set_variable(real_feat_temp_value, r=True)
                tmp_prop_feat_calculator_in_list.append(init_real_calculator)
                real_prop_feat_list.append(init_real_calculator.value)

                imag_feat_temp_value = ada_platform_one_step_propagation(self.imag_adj, imag_prop_feat_list[-1])
                init_imag_calculator.set_variable(imag_feat_temp_value, i=True)
                tmp_prop_feat_calculator_in_list.append(init_imag_calculator)
                imag_prop_feat_list.append(init_imag_calculator.value)

            
            else:
                for k in range(len(tmp_prop_feat_calculator_in_list)):
                    tmp_calculator = tmp_prop_feat_calculator_in_list[k]
                    new_calculator = calculator(tmp_calculator.value, tmp_calculator.r_step, tmp_calculator.i_step)
                    tmp_value = ada_platform_one_step_propagation(self.real_adj, tmp_calculator.value)
                    new_calculator.set_variable(tmp_value, r=True)
                    tmp_prop_feat_calculator_out_list.append(new_calculator)

                for k in range(len(tmp_prop_feat_calculator_in_list)):
                    tmp_calculator = tmp_prop_feat_calculator_in_list[k]
                    new_calculator = calculator(tmp_calculator.value, tmp_calculator.r_step, tmp_calculator.i_step)
                    tmp_value = ada_platform_one_step_propagation(self.imag_adj, tmp_calculator.value)
                    new_calculator.set_variable(tmp_value, i=True)
                    new_calculator.reversal()
                    tmp_prop_feat_calculator_out_list.append(new_calculator)

                real_feat, imag_feat = calculate_real_imag_feat(tmp_prop_feat_calculator_out_list)
                real_prop_feat_list.append(real_feat)
                imag_prop_feat_list.append(imag_feat)
                tmp_prop_feat_calculator_in_list = tmp_prop_feat_calculator_out_list
                tmp_prop_feat_calculator_out_list = []

        return [torch.FloatTensor(feat) for feat in real_prop_feat_list], [torch.FloatTensor(feat) for feat in imag_prop_feat_list]


# Might include training parameters
class ComMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(ComMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, real_feat_list, imag_feat_list):
        return NotImplementedError

    def aggregate(self, real_feat_list, imag_feat_list):
        if not isinstance(real_feat_list, list) or not isinstance(imag_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in real_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The real feature matrices must be tensors!")
        for feat in imag_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The imag feature matrices must be tensors!")
            
        return self.combine(real_feat_list, imag_feat_list)


class TwoDirGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.un_adj = None
        self.in_adj = None
        self.out_adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.un_adj, self.in_adj, self.out_adj = self.construct_adj(adj)
        un_prop_feat_list = []
        in_prop_feat_list = []
        out_prop_feat_list = []

        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.un_adj.shape[1] != feature.shape[0] or self.in_adj.shape[1] != feature.shape[0] or self.out_adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")

        un_prop_feat_list = [feature]
        in_prop_feat_list = [feature]
        out_prop_feat_list = [feature]

        for _ in range(self.prop_steps):
            un_feat_temp = ada_platform_one_step_propagation(self.un_adj, un_prop_feat_list[-1])
            in_feat_temp = ada_platform_one_step_propagation(self.in_adj, in_prop_feat_list[-1])
            out_feat_temp = ada_platform_one_step_propagation(self.out_adj, out_prop_feat_list[-1])
            un_prop_feat_list.append(un_feat_temp)
            in_prop_feat_list.append(in_feat_temp)
            out_prop_feat_list.append(out_feat_temp)

        return [torch.FloatTensor(feat) for feat in un_prop_feat_list], \
            [torch.FloatTensor(feat) for feat in in_prop_feat_list], \
                [torch.FloatTensor(feat) for feat in out_prop_feat_list]


# Might include training parameters
class TwoDirMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(TwoDirMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, un_feat_list, in_feat_list, out_feat_list):
        return NotImplementedError

    def aggregate(self, un_feat_list, in_feat_list, out_feat_list):
        if not isinstance(un_feat_list, list) or not isinstance(in_feat_list, list) or not isinstance(out_feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in un_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The un direction feature matrices must be tensors!")
        for feat in in_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The in direction feature matrices must be tensors!")
        for feat in out_feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The out direction feature matrices must be tensors!")
            
        return self.combine(un_feat_list, in_feat_list, out_feat_list)
    

def ada_platform_one_step_propagation(adj, x):
    if platform.system() == "Linux":
        one_step_prop_x = csr_sparse_dense_matmul(adj, x)
    else:
        one_step_prop_x = adj.dot(x)
    return one_step_prop_x

def calculate_real_imag_feat(tmp_prop_feat_calculator_out_list):
    real_feat_list = []
    imag_feat_list = []
        
    for k in range(len(tmp_prop_feat_calculator_out_list)):
        tmp_calculator = tmp_prop_feat_calculator_out_list[k]
        if tmp_calculator.i_step & 1 == 0 and tmp_calculator.i_step != 0:
            real_feat_list.append(tmp_calculator.value)
        elif tmp_calculator.i_step == 0:
            real_feat_list.append(tmp_calculator.value)
        else:
            imag_feat_list.append(tmp_calculator.value)

    if len(real_feat_list) != len(imag_feat_list):
        raise RuntimeError("Something wrong!")
    

    for k in range(len(real_feat_list)):
        if k == 0:
            real_feat = real_feat_list[k]
            imag_feat = imag_feat_list[k]
        else:
            real_feat += real_feat_list[k]
            imag_feat += imag_feat_list[k]
    return real_feat, imag_feat