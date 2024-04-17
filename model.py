import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv

class robustMLP(nn.Module):
    def __init__(self, nfeat, nh1, nclass, dropout, bias=True):
        super(robustMLP, self).__init__()
        self.w0 = Parameter(torch.FloatTensor(nfeat, nh1))
        self.w1 = Parameter(torch.FloatTensor(nh1, nclass))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(nclass))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w0, gain=1.414)
        nn.init.xavier_uniform_(self.w1, gain=1.414)
        
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, input):
        x = torch.mm(input, self.w0)
        x1 = self.relu(x)
        x = self.dropout(x1)
        x = torch.mm(x, self.w1)
        if self.bias is not None:
            output = x + self.bias
        return F.normalize(x1), F.log_softmax(output, dim=1)
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels,hidden_channels,num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads,hidden_channels,num_heads))
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, 1))
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
    
class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 ):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=2, cached=True)
        self.reset_parameters()
    def reset_parameters(self):
        self.conv.reset_parameters()
    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        return x.log_softmax(dim=-1)
    
class SIGN(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 ):
        super(SIGN, self).__init__()
        self.conv = []
        for _ in range(4):
            self.conv.append(SGConv(in_channels, out_channels, K=2, cached=True))
        self.out_conv = nn.Linear(4 * out_channels, out_channels)
        self.reset_parameters()
    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
    def forward(self, x, adj_t):
        x_temp = []
        for i in range(4):
            x_temp.append(self.conv[i](x, adj_t))
        x_out = torch.cat(x_temp, dim=1)
        x_out = self.out_conv(x_out)
        return x_out.log_softmax(dim=-1)
