import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math
from utils import *
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor
import pandas as pd

warnings.filterwarnings("ignore", category=Warning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjacency, input_feature, similarity, adj, bi_adj, output_sl):

        new_bi = torch.matmul(adj.to(device), bi_adj.to(device))
        new_bi = F.normalize(new_bi, p=1, dim=1)
        h_matrix = torch.matmul(adjacency, new_bi)
        prob_label = torch.matmul(new_bi, output_sl)

        support = torch.matmul(input_feature, h_matrix)
        output = torch.matmul(support, self.weight)

        if self.use_bias:
            output += self.bias
        return output, h_matrix, prob_label


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv1 = GraphConvolution(input_dim, hidden_dim, False)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim, False)

        self.fc1 = nn.Linear(116 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, adjacency, input_feature, similarity, labels_mlp, adj, bi_adj, output_sl):
        gcn1, a1, prob_label = self.conv1.forward(adjacency, input_feature, similarity, adj, bi_adj, output_sl)
        gcn1, a1 = F.relu(gcn1), F.relu(a1)
        gcn2, a2, prob_label = self.conv2.forward(a1, gcn1, similarity, adj, bi_adj, output_sl)
        # gcn2, a2, prob_label = self.conv2.forward(adjacency, gcn1, similarity, adj, bi_adj, output_sl)
        gcn2 = F.relu(gcn2)

        gcn2_flatten = gcn2.reshape(adjacency.shape[0], -1)
        gcn2_flatten = self.dropout(gcn2_flatten)
        fc1 = F.relu(self.fc1(gcn2_flatten))
        fc2 = self.fc2(fc1)
        logits = self.fc3(fc2)
        fc3_value = [fc1, fc2, logits]

        logits = logits + labels_mlp
        return F.log_softmax(logits), fc3_value, prob_label


class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass, dropout):
        super(MLP, self).__init__()
        self.n_feat = n_feat
        self.n_hid = n_hid
        self.n_class = nclass

        self.fc1 = nn.Linear(n_feat*n_feat, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid//2)
        self.fc3 = nn.Linear(n_hid//2, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        inputFeature = x.reshape(x.shape[0], -1)
        inputFeature = self.dropout(inputFeature)
        l1 = F.relu(self.fc1(inputFeature))
        l2 = self.fc2(l1)
        l3 = self.fc3(l2)

        return F.log_softmax(l3)

    def get_emb(self, x):
        return self.mlp[0](x).detach()


class MLP_Label(nn.Module):
    def __init__(self, n_feat, n_hid, nclass, dropout=0.3):
        super(MLP_Label, self).__init__()
        self.n_feat = n_feat
        self.n_hid = n_hid
        self.n_class = nclass

        self.fc1 = nn.Linear(n_feat, n_hid)
        self.fc2 = nn.Linear(n_hid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l1 = self.dropout(l1)
        l2 = self.fc2(l1)
        return F.log_softmax(l2)

    def get_emb(self, x):
        return self.mlp[0](x).detach()