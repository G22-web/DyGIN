import math

import torch
import numpy as np
import utils as u
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907-> 2017 ICLR "Semi-Supervised Classification with Graph Convolutional Networks"
    """

    def __init__(self, device, in_features, out_features, time_step, time_weight, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.time_step = time_step
        self.time_weight = time_weight
        self.out_features = out_features
        self.device = device
        # self.rows = in_features
        # self.cols = out_features

        cell_args = u.Namespace({})
        cell_args.rows = in_features
        cell_args.cols = out_features

        # LSTM
        self.evolve_weights = mat_LSTM_cell(cell_args,device)

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.time_step>=1:
            self.GCN_weights = self.evolve_weights(self.time_weight, device) # 调用mat_LSTM_cell
            self.dgcn_weight = self.GCN_weights
            self.dgcn_weight = Parameter(self.dgcn_weight) #Parameter:默认有梯度,将一个不可训练的tensor转换成可以训练的类型parameter


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, input, adj):
        if self.time_step >= 1:
            # 8.使用LSTM更新GCN的权重参数
            self.first_weight = self.dgcn_weight
            # print("--------self.first_weight-------------")
            # print(self.first_weight.shape) #[136, 90]
            # print(input.shape) #[8704, 90]
            input = input.reshape(-1, self.first_weight.shape[0]) #[5760,136]
            support = torch.mm(input, self.first_weight)
            support = support.transpose(0, 1)
            output = torch.spmm(adj, support) #[90, *]
        else:
            self.first_weight = self.weight #[136, 90]
            input = input.reshape(-1, self.first_weight.shape[0])
            support = torch.mm(input, self.first_weight) # tensor维度必须为2,[a,b] [b,c] -> [a,c] [5760,90]
            support = support.transpose(0, 1)
            adj = adj.to(torch.float32) #[90,90]
            output = torch.spmm(adj, support) # 矩阵乘法 [90,*]
        # print(self.bias.shape) # 128
        if self.bias is not None:
            return output + self.bias, self.first_weight
        else:
            return output, self.first_weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class mat_LSTM_cell(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args  ##arg.rows = in_feats; arg.cols= out_feats
        # print(args.rows) # 90 即features.shape[1]
        # print(args.cols) # 90 即args.hidden
        self.update = mat_LSTM_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid().to(device), device)

        self.reset = mat_LSTM_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid().to(device), device)

        self.htilda = mat_LSTM_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh().to(device), device)

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q,device):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        prev_Q = prev_Q.to(device)
        prev_Q = prev_Q
        z_topk = prev_Q
        # print("--------mat_LSTM_cell-----------")
        # print(prev_Q)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_LSTM_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation, device):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows)).to(device)
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows)).to(device)
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols)).to(device)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        x = x
        hidden = hidden
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)
        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
