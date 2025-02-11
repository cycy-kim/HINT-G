import torch
import torch.nn as nn
import torch.nn.functional as F

from numbers import Number

from .layers import GraphConvolution_Graph, GraphConvolution_Node

class GCN_Node(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GCN_Node, self).__init__()
        self.use_bias = args.bias
        self.bn = args.bn
        self.args = args

        try:
            hiddens = [int(s) for s in self.args.hiddens.split('-')]
        except:
            hiddens = [self.args.hidden1]
        
        if self.bn:
            self.bnlayers = nn.ModuleList([nn.BatchNorm1d(num_features=hidden) for hidden in hiddens[:-1]])

        self.layers_ = nn.ModuleList()
        layer0 = GraphConvolution_Node(args=args, input_dim=input_dim, output_dim=hiddens[0], activation=F.relu, bias=self.use_bias)
        self.layers_.append(layer0)

        for i in range(1, len(hiddens)):
            layertemp = GraphConvolution_Node(args=args, input_dim=hiddens[i-1], output_dim=hiddens[i], activation=F.relu, bias=self.use_bias)
            self.layers_.append(layertemp)

        # self.pred_layer = nn.Linear(hiddens[-1], output_dim)
        self.pred_layer = nn.Linear(sum(hiddens), output_dim)
        self.hiddens = hiddens

    def forward(self, inputs, training=None):
        emb = self.embedding(inputs, training)
        x = self.pred_layer(emb)
        return x

    def embedding(self, inputs, training=None):
        """_summary_

        Args:
            x: [num_node, num_features]
            support: [num_node, num_node]

        Returns:
            if args.concat:
                x: [num_node, sum(hiddens)]
            if not args.concat:
                x: [num_node, hiddens[-1]]
        """
        x, support = inputs
        nmb_layers = len(self.layers_)
        x_all = []

        for layer_index in range(nmb_layers):
            x = self.layers_[layer_index]((x, support))
            if self.bn and layer_index != nmb_layers - 1:
                x = self.bnlayers[layer_index](x)
            x_all.append(x)
        
        if self.args.concat:
            x = torch.cat(x_all, dim=-1)
        else:
            x = x_all[-1]

        return x

class GCN_Graph(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GCN_Graph, self).__init__()
        self.args = args
        usebias = args.bias
        self.bn = args.bn

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens = [args.hidden1]

        self.layers_ = nn.ModuleList()
        self.bnlayers = nn.ModuleList()

        layer0 = GraphConvolution_Graph(args=args, input_dim=input_dim, output_dim=hiddens[0], activation=F.relu, bias=usebias)
        self.layers_.append(layer0)

        for i in range(1, len(hiddens)):
            layertemp = GraphConvolution_Graph(args=args, input_dim=hiddens[i-1], output_dim=hiddens[i], activation=F.relu, bias=usebias)
            self.layers_.append(layertemp)
            if self.bn:
                self.bnlayers.append(nn.BatchNorm1d(num_features=hiddens[i]))

        self.pred_layer = nn.Linear(hiddens[-1] * 2, output_dim)

    def forward(self, inputs, training=None):
        x, support = inputs
        x = self.getNodeEmb((x, support), training)

        out1 = torch.max(x, dim=1)[0]
        out2 = torch.sum(x, dim=1)
        out = torch.cat([out1, out2], dim=-1)

        out = self.pred_layer(out)
        return out

    def getNodeEmb(self, inputs, training=None):
        x, support = inputs
        x_all = []

        for layerindex, layer in enumerate(self.layers_):
            x = layer((x, support))
            if self.bn and layerindex != len(self.layers_) - 1:
                x = self.bnlayers[layerindex](x)
            x_all.append(x)

        if self.args.concat:
            x = torch.cat(x_all, dim=-1)
        else:
            x = x_all[-1]

        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        """MLP implementation for GraphCFE(CLEAR)

        Args:
            input_dim (_type_): _description_
            output_dim (_type_): _description_
            hidden_dim (_type_): _description_
            n_layers (_type_): _description_
            activation (str, optional): _description_. Defaults to 'none'.
            slope (float, optional): _description_. Defaults to .1.
            device (str, optional): _description_. Defaults to 'cpu'.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
