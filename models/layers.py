import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Linear
from torch_sparse import spmm

from utils import dtype

class GraphConvolution_Node(nn.Module):
    """Graph convolution layer."""
    def __init__(self, args, input_dim, output_dim, activation=F.relu, bias=False):
        super(GraphConvolution_Node, self).__init__()
        self.args = args
        self.activation = activation
        self.bias = bias

        # Initialize weights with He or Glorot normalization based on args.initializer
        if self.args.initializer == 'he':
            self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(2. / input_dim))
        else:  # Glorot initialization
            stdv = torch.sqrt(torch.tensor(2. / (input_dim + output_dim)))
            self.weight = nn.Parameter(torch.rand(input_dim, output_dim) * 2 * stdv - stdv)

        if self.bias:
            self.bias_weight = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs, training=None):
        x, support = inputs

        if training and self.args.dropout > 0:
            x = F.dropout(x, p=self.args.dropout, training=training)

        if self.args.order == 'AW':
            if support.is_sparse:
                output = torch.sparse.mm(support, x)
            else:
                output = torch.matmul(support, x)
            output = torch.matmul(output, self.weight)
        else:
            pre_support = torch.matmul(x, self.weight)
            if support.is_sparse:
                output = torch.sparse.mm(support, pre_support)
            else:
                output = torch.matmul(support, pre_support)

        if self.bias:
            output += self.bias_weight

        if self.args.embnormalize:
            output = F.normalize(output, p=2, dim=1)

        return self.activation(output)


class GraphConvolution_Graph(nn.Module):
    """Graph convolution layer."""
    def __init__(self, args, input_dim, output_dim, activation=F.relu, bias=False):
        super(GraphConvolution_Graph, self).__init__()
        self.args = args
        self.activation = activation
        self.bias = bias
        
        if args.initializer == 'he':
            self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(2. / input_dim))).to(dtype)
        else:
            self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(2. / (input_dim + output_dim)))).to(dtype)
        
        if self.bias:
            self.bias_weight = nn.Parameter(torch.zeros(output_dim)).to(dtype)

    def forward(self, inputs):
        x, support = inputs
        x = x.to(self.weight.device)
        support = support.to(self.weight.device)
        if self.args.dropout > 0:
            x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        if self.args.order == 'AW':
            if support.is_sparse:
                support_x = torch.sparse.mm(support, x)
            else:
                support_x = torch.matmul(support.float(), x)
            output = torch.matmul(support_x, self.weight)
        else:
            pre_sup = torch.matmul(x, self.weight)
            if support.is_sparse:
                output = torch.sparse.matmul(support, pre_sup)
            else:
                output = torch.matmul(support, pre_sup)

        if self.bias:
            output += self.bias_weight
        
        if self.args.embnormalize:
            output = F.normalize(output, p=2, dim=-1)

        return self.activation(output)
