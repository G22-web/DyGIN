import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import math
import utils as u
from torch.nn import init
from scipy.linalg import fractional_matrix_power
sys.path.append("models/")
from mlp import MLP
from layers_RML import GraphConvolution



def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def Comp_degree(A):
    """ compute degree matrix of a graph """
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    diag = torch.eye(A.size()[0]).cuda()

    degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A))

    return degree_matrix


class GraphConv_kipf(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_kipf, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.weight = nn.Parameter(
        #         torch.FloatTensor(in_dim, out_dim))
        # self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        # init.xavier_uniform_(self.weight)
        # init.constant_(self.bias, 0)

        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)


    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        A_norm = A + torch.eye(n).cuda()
        deg_mat = Comp_degree(A_norm)
        frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.cpu(),
                                                                -0.5)).cuda()
        # frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.cpu().detach().numpy(),
        #                                                         -0.5)).cuda()

        A_hat = torch.matmul(torch.matmul(frac_degree
            , A_norm), frac_degree)


        repeated_A =A_hat.repeat(b,1,1)
        agg_feats = torch.bmm(repeated_A, features)
    
        # out = torch.einsum('bnd,df->bnf', (agg_feats, self.weight))
        # out = out + self.bias
        out = self.MLP(agg_feats.view(-1, d))
        out = self.batchnorm(out).view(b, -1, self.out_dim)

        return out 

class GraphConv(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim #[136]
        self.out_dim = out_dim #[128]
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A, t):
        b, n, d = features.shape  # torch.Size([64, 90, 136])
        assert (d == self.in_dim)  # [136]

        # print("--------MLP---------")
        # print(A.shape) #[90,90]
        repeated_A = A.repeat(b, 1, 1)  # #torch.Size([64, 90, 90])

        '''包含MLP'''
        repeated_A = torch.tensor(repeated_A, dtype=torch.float32)
        agg_feats = torch.bmm(repeated_A,
                              features)  # [64,90,136]torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        out = self.MLP(agg_feats.view(-1, d))
        out = F.relu(self.batchnorm(out)).view(b, -1, self.out_dim)  # [64,90,64]

        return out



class Inception_layer(nn.Module):
    def __init__(self, input_dim):
        super(Inception_layer, self).__init__()
        self.input_dim = input_dim
        self.num_layers = 2
        '''baseline'''
        self.GCN_1 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 128) / 4), 128)
        self.GCN_2 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 64) / 4), 64)

        '''增加小批次的GCN'''
        # self.GCN = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 32) / 4), 32)
        # self.GCN_1 = GraphConv(self.num_layers, 32, int((self.input_dim + 128) / 4), 128)
        # self.GCN_2 = GraphConv(self.num_layers, 32, int((self.input_dim + 64) / 4), 64)
    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1))])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def forward(self, A, h, padded_neighbor_list, t):
        b, c, d = h.shape
        '''baseline'''
        out_1 = self.maxpool(h.view(-1, d), padded_neighbor_list).view(h.shape)
        out_2 = self.GCN_1(h, A, t)
        out_3 = self.GCN_2(h, A, t)
        out = torch.cat((out_1, out_2, out_3), dim=2)

        '''增加小批次的GCN'''
        # out_1 = self.maxpool(h.view(-1, d), padded_neighbor_list).view(h.shape)  # torch.Size([128, 90, 136])
        # # print("-----0-----")
        # # print(out_1.shape)
        # out_1_1 = self.GCN(out_1, A)  # torch.Size([128, 90, 32])
        # # print("-----1-----")
        # # print(out_1_1.shape)
        # out_2_1 = self.GCN(h, A) # torch.Size([128, 90, 32])
        # # print("----2------")
        # # print(out_2_1.shape)
        # out_2 = self.GCN_1(out_2_1, A)  # torch.Size([128, 90, 128])
        # # print("----3------")
        # # print(out_2.shape)
        # out_3_1 = self.GCN(h, A) # torch.Size([128, 90, 32])
        # # print("----4------")
        # # print(out_3_1.shape)
        # out_3 = self.GCN_2(out_3_1, A)  # torch.Size([128, 90, 64])
        # # print("----5------")
        # # print(out_3.shape)
        #
        # out = torch.cat((out_1_1, out_2, out_3),dim=2)  # torch.Size([128, 90, 328]) torch.cat:除拼接维数dim数值可不同外其余维数数值需相同，方能对齐

        return out


class GraphConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphConvolutionalAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        encoded_output, _ = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)

        return decoded_output

class Graph_Inception(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, final_dropout,
                 device, dataset,batch_size, num_nodes, A, t, num_hid, A_adj, time_weight, readout="average", corruption="node_shuffle", bias=False):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            device: which device to use
        '''

        super(Graph_Inception, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.dataset = dataset
        self.batch_size = batch_size
        self.time_step = t
        self.time_weight = time_weight


        ###Adj matrix
        # self.Adj_ = torch.nn.Parameter(torch.rand([num_nodes,num_nodes]), requires_grad=True) # num_nodes * num_nodes维的张量; requires_grad默认值为True，表示可训练，False表示不可训练。
        self.Adj = torch.nn.Parameter(torch.rand([num_nodes,num_nodes]), requires_grad=True)

        ###Pool matrix
        self.Pool = torch.nn.Parameter(torch.ones(size=([num_nodes])), requires_grad=True) # torch.ones:返回一个全为1 的张量，形状由可变参数sizes定义

        self.readout = getattr(self, "_%s" % readout)
        self.corruption = getattr(self, "_%s" % corruption)

        ###List of Inception layers
        self.Inceptions = torch.nn.ModuleList()
        c = 0
        # print("---------------input_dim:---------------------")
        # print(input_dim) # 136
        for i in range(self.num_layers):
            self.Inceptions.append(Inception_layer(input_dim+c))
            c += 192

        cell_args = u.Namespace({})
        cell_args.rows = input_dim  # 136
        cell_args.cols = input_dim

        # cell_args = u.Namespace({})
        # cell_args.rows = input_dim  # 136
        # cell_args.cols = num_hid


        # LSTM
        self.evolve_weights = mat_LSTM_cell(cell_args, device) # GRU(W_t-1)
        # self.evolve_weights = mat_LSTM_cell_2(cell_args, device)  # GRU(H_t-1, W_t-1)、GRU(H_t-1, A_t-1)

        # self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, num_hid), requires_grad=True)
        self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, input_dim), requires_grad=True)

        self.Linear = nn.Linear(136, 136).to(device)
        self.gru = nn.GRUCell(136, 136).to(device)  # 使用GRU进行权重更新

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=136,  # input feature (frequency) dim after maxpooling 128*563 -> 64*140 (freq*time)
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=1024,  # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
            dropout=0.5,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        ).to(device)
        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4).to(device)

        self.AutoModel = GraphConvolutionalAutoencoder(input_dim, input_dim)

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


        if self.time_step>=1:
            # 调用Transformer
            self.time_weight = self.time_weight.reshape(1, 136, 136)
            self.GCN_weights = self.transformer_encoder(self.time_weight)
            self.GCN_weights = self.GCN_weights.reshape(136, 136)
            # print("---------self.time_weight----------")
            # print(self.time_weight.shape) #[136,136]
            # self.GCN_weights = self.evolve_weights(self.time_weight, device) # 调用mat_LSTM_cell
            # self.GCN_weights = self.gru(self.time_weight, self.time_weight) # 调用GRU
            # self.GCN_weights = self.Linear(self.time_weight)



            self.dgcn_weight = self.GCN_weights
            self.dgcn_weight = torch.nn.Parameter(self.dgcn_weight,  requires_grad=True) #Parameter:默认有梯度,将一个不可训练的tensor转换成可以训练的类型parameter



            ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(input_dim, affine=False)
        

        #Linear functions that maps the hidden representations to labels
        '''baseline: Pooling层'''
        # self.classifier = nn.Sequential(
        #                     nn.Linear((c+input_dim)*3, 512),
        #                     nn.Dropout(p=self.final_dropout),
        #                     nn.PReLU(512),
        #                     nn.Linear(512, output_dim))
        '''max+min(h)+mean'''
        # self.classifier = nn.Sequential(
        #     nn.Linear(1041, 512),
        #     nn.Dropout(p=self.final_dropout),
        #     nn.PReLU(512),
        #     nn.Linear(512, output_dim))
        '''max、mean、min、len'''
        # self.classifier = nn.Sequential(
        #     nn.Linear(520, 512),
        #     nn.Dropout(p=self.final_dropout),
        #     nn.PReLU(512),
        #     nn.Linear(512, output_dim))
        '''len+mean'''
        self.classifier = nn.Sequential(
            nn.Linear(1040, 512),
            # layer=0,(272,512); layer=1,(656,512);   layer=2,(1040, 512);  layer=3,(1424,512); layer=4,(1808,512);
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(512),
            nn.Linear(512, output_dim))
    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, batch_graph, A):
        X_concat = torch.cat([graph.node_features.view(1,-1,graph.node_features.shape[1]) for graph in batch_graph], 0).to(self.device)
        # A = self.Adj_ # 可学习的

        B, N, D = X_concat.shape # 64, 90, 136

        X_concat = X_concat.view(-1, D)
        X_concat = self.bn0(X_concat)
        X_concat = X_concat.view(B, N, D)


        padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph) # [5760, 2]


        h = X_concat #[64, 90, 136]

        ''' LSTM 捕获动态图的时间信息'''
        if self.time_step >= 1:

            '''GRU(X_t-1,W_t-1)'''
            # GCN_weights = self.evolve_weights(self.time_weight, h, self.device)  # 调用mat_LSTM_cell
            # dgcn_weight = GCN_weights
            # dgcn_weight = torch.nn.Parameter(dgcn_weight,  requires_grad=True) #[136,5760] Parameter:默认有梯度,将一个不可训练的tensor转换成可以训练的类型parameter
            # self.first_weight = dgcn_weight  # [136, 5760]
            # print(self.first_weight.shape)
            # h = h.reshape(-1,136)
            # h = torch.mm(h, self.first_weight) # [136 5760]
            # h = h.view(B, N, D)  # [64, 90, 136]
            #
            # for layer in self.Inceptions:
            #     h = layer(A, h, padded_neighbor_list, self.time_step)

            '''使用LSTM (W_t-1) or GRU 更新GCN的权重参数'''
            h = h.reshape(-1, 136)  #  [5760, 136]
            self.first_weight = self.dgcn_weight #[136, 136]
            # first_weight = self.AutoModel(X_concat) # LSTM自编码器：参数必须是三维的
            # print("-----first_weight-----------")
            # print(first_weight.shape)

            h = torch.mm(h, self.first_weight) #[5760 136]
            h = h.view(B, N, D) #[64, 90, 136]
            # Inception 层
            for layer in self.Inceptions:
                h = layer(A, h, padded_neighbor_list,self.time_step)
        else:
            h = h.reshape(-1, 136) #[5760, 136]
            self.first_weight = self.weight #[136,136]
            h = torch.mm(h, self.first_weight)  # tensor维度必须为2,[a,b] [b,c] -> [a,c] #[5760, 136]
            h = h.view(B, N, D) #[64, 90, 136]

            for layer in self.Inceptions:
                h = layer(A, h, padded_neighbor_list, self.time_step)

        # 不添加LSTM
        # self.first_weight = self.weight
        # for layer in self.Inceptions:
        #     h = layer(A, h, padded_neighbor_list)



        # 池化层
        max_pool,ind = torch.max(h,dim=1) #[64, 520]
        min_pool = torch.min(h,dim=1)[0] #[64, 520]
        mean_pool = torch.mean(h,dim=1) #[64, 520]
        sum_pool= torch.sum(h, dim=1)
        '''baseline: Pooling层'''
        # repeated_pool = self.Pool.repeat(h.shape[0], 1, 1) # repeat():沿着指定的维度重复tensor的数据
        # weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1) # [64, 520] torch.bmm:两个tensor的矩阵乘法； view():重新调整Tensor的形状
        # pooled = torch.cat((max_pool, weighted_pool, mean_pool), dim=1) # [64, 1560] 将tensor类型拼接起来

        '''len+mean (目前)'''
        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)  # repeat():沿着指定的维度重复tensor的数据
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0], -1)  # [64, 520] torch.bmm:两个tensor的矩阵乘法； view():重新调整Tensor的形状
        pooled = torch.cat((weighted_pool, mean_pool), dim=1)  #[64, 1040] 将tensor类型拼接起来

        '''max+min(h)+mean'''
        # repeated_pool = self.Pool.repeat(h.shape[0], 1, 1) # repeat():沿着指定的维度重复tensor的数据
        # weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1) # [128,50]; torch.bmm:两个tensor的矩阵乘法； view():重新调整Tensor的形状
        # weighted_pool = torch.min(weighted_pool,dim=1)[0] # torch.Size([128])
        # weighted_pool = weighted_pool.reshape(-1, 1) # torch.Size([128,1])
        # pooled = torch.cat((max_pool, weighted_pool, mean_pool), dim=1) # [128,1041] 将tensor类型拼接起来

        # print("---------pooled---------")
        # print(pooled.shape)

        score = self.classifier(pooled)
        return score, self.first_weight

    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A, edge):
        if edge:  # 边的打乱
            perm_A = torch.randperm(A.size(0))  # 随机打乱后获得的数字序列
            neg_A = A[perm_A]
            A = neg_A
        else:  # 节点特征的打乱
            # print("-----A-----")
            # print(X.size(0)) #128
            # print(X[torch.randperm(X.size(0))].shape)#[128,90,136]
            perm = torch.randperm(X.size(0))  # 随机打乱后获得的数字序列
            neg_X = X[perm]
            X = neg_X
            # print(X.shape) #[128,90,136]
        return X, A

class mat_LSTM_cell_2(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args  ##arg.rows = in_feats; arg.cols= out_feats
        # print("-------mat_LSTM_cell------------")
        # print(args.rows) # 136
        # print(args.cols) # 136
        self.update = mat_LSTM_gate_2(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid().to(device), device)

        self.reset = mat_LSTM_gate_2(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid().to(device), device)

        self.htilda = mat_LSTM_gate_2(args.rows,
                                   args.cols,
                                   torch.nn.Tanh().to(device), device)

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q, prev_Z, device):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z) #[136, 12240]
        z_topk = prev_Z  #[64,90,136] 特征矩阵 Xt-1
        prev_Q = prev_Q.to(device) #[136,136] Wt-1

        update = self.update(z_topk, prev_Q) # [136,5760]
        reset = self.reset(z_topk, prev_Q) # [136,5760]


        h_cap = reset * prev_Q #[136, 5760]
        # h_cap = h_cap.reshape(B,N)
        h_cap = self.htilda(z_topk, h_cap) # [136,680]

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_LSTM_gate_2(torch.nn.Module):
    def __init__(self, rows, cols, activation, device):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device) #[136,136]  torch.nn.Parameter:将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.reset_param(self.W)

        self.U = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device) #[136,136]
        self.reset_param(self.U)

        self.bias = torch.nn.Parameter(torch.zeros(rows, cols)).to(device) #[136,136]

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        x = x #ht-1:[64,90,136],
        hidden = hidden #Wt-1:[136,136],
        x = x.reshape(136, -1)



        # print("------2---------")
        # # print(self.W.shape) #[136,136]
        # # print(self.U.shape) #[136,136]
        # print(self.W.matmul(x).shape) #[136,5760]
        # print(self.U.matmul(hidden).shape) #[136,5760]
        # print(self.bias.shape)#[136,5760]

        x = x.reshape(136, -1)
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)# 维度不同，调试不通
        # out = self.activation(self.W.matmul(x) + self.U.matmul(hidden))  # 维度不同，调试不通

        return out


class mat_LSTM_cell(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args  ##arg.rows = in_feats; arg.cols= out_feats
        # print("-------mat_LSTM_cell------------")
        # print(args.rows) # 136
        # print(args.cols) # 136
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

    def forward(self, prev_Q, device):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        prev_Q = prev_Q.to(device)  # [680, 136]
        prev_Q = prev_Q
        z_topk = prev_Q  # [64,90,136]

        update = self.update(z_topk, prev_Q)  # [136,680]
        reset = self.reset(z_topk, prev_Q)  # [136,680]

        h_cap = reset * prev_Q  # [136, 680]
        h_cap = h_cap.transpose(0, 1)
        h_cap = self.htilda(z_topk, h_cap)  # [136,680]

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_LSTM_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation, device):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device)  # [136,136]  torch.nn.Parameter:将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 不进行更新
        # self.W = torch.Tensor(rows, rows).to(device)  # [136,136]  torch.nn.Parameter:将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.reset_param(self.W)

        self.U = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device)  # [136,136]
        # 不进行更新
        # self.U = torch.Tensor(rows, rows).to(device)  # [136,136]
        self.reset_param(self.U)

        self.bias = torch.nn.Parameter(torch.zeros(rows, cols)).to(device)
        # 不进行更新
        # self.bias = torch.zeros(rows, cols).to(device)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        x = x  #
        hidden = hidden  # [136,136]

        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)  # 单向GRU
        return out

class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = torch.nn.Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs): # ,mask
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        # scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        # print("------------------")
        # print( node_embs[topk_indices].shape) #[136 90, 136]
        # print(tanh(scores[topk_indices].view(-1, 1)).shape) #[12240, 1]
        out = node_embs[topk_indices].view(-1, 136) * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()

class Graph_CNN_kipf(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device, dataset,batch_size):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(Graph_CNN_kipf, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.dataset = dataset
        self.batch_size = batch_size

        ###Adj matrix
        self.Adj = torch.nn.Parameter(torch.rand([90,90]), requires_grad=False)

        ###Pool matrix
        self.Pool = torch.nn.Parameter(torch.ones(size=([90])), requires_grad=False)

        ###List of GCN layers
        self.GCN_1 = GraphConv_kipf(input_dim, 128)
        self.GCN_2 = GraphConv_kipf(128, 128)

        ###List of pooling layers

        ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(136, affine=False)
        

        #Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
                            nn.Linear(1*128, 512),
                            nn.Dropout(p=self.final_dropout),
                            nn.PReLU(512),
                            nn.Linear(512, output_dim))



    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1,-1,136) for graph in batch_graph], 0).to(self.device)
        # A = F.relu(self.Adj)
        
        A = np.zeros([90, 90])
        Num_hop = 1
        for i in range(A.shape[0]):
            # A[i, i] = 1
            for j in range(A.shape[0]):
                if (i - j <= Num_hop) and (i - j > 0):
                    A[i, j] = 1
                    A[j, i] = 1
        A = torch.FloatTensor(A).to(self.device)

        B, N, D = X_concat.shape

        # X_concat = X_concat.view(-1, D)
        # X_concat = self.bn0(X_concat)
        # X_concat = X_concat.view(B, N, D)

        h = F.relu(self.GCN_1(X_concat, A))
        h = F.softmax(self.GCN_2(h, A))

        # max_pool = torch.max(h,dim=1)[0]
        # min_pool = torch.min(h,dim=1)[0]
        # mean_pool = torch.mean(h,dim=1)
        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1)
        # pooled = torch.cat((max_pool, min_pool, mean_pool), dim=1)
        pooled = weighted_pool

        score = self.classifier(pooled)

        return score

class Discriminator(nn.Module):
    def __init__(self, n_h, device):
        super(Discriminator, self).__init__()
        """
        卷积神经网络中的全连接层需要调用nn.Linear就可以实现
        """
        self.net = nn.Bilinear(n_h, n_h, 1).to(device) # similar to score of CPC 与CPC分数相近，要求参数需是三维的

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, H, neg_H, neg_X):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param H: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        # g = torch.unsqueeze(g, dim=1) #[128,1,128]
        # H = H.reshape(g.shape[0], -1, g.shape[0]) #[128,90,128]
        # neg_H = neg_H.reshape(g.shape[0], -1, g.shape[0])  # [128,90,128]
        # neg_X = neg_X.reshape(g.shape[0], -1, g.shape[0])  # [128,90,128]
        # g = g.expand_as(H).contiguous()  # 将s的维度指定为h_rl的维度，并将内存变成连续的 [128,90,128]

        H = H.transpose(0,1) #[17408, 90]
        neg_H = neg_H.transpose(0, 1) #[17408, 90]
        neg_X = neg_X.transpose(0, 1) #[17408, 90]
        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(H, g), dim=0) # [17408, 1]
        sc_H_fk = torch.squeeze(self.net(neg_H, g), dim=0) # [17408, 1]
        sc_X_fk = torch.squeeze(self.net(neg_X, g), dim=0) # [17408, 1]


        logits = torch.cat((sc_rl, sc_H_fk, sc_X_fk), dim=1) #[17408, 3]
        logits = logits.reshape(128, -1)
        return logits

class DGCN(nn.Module):
    def __init__(self, device, num_nodes, batch_size, num_classes, A, num_feat, num_hid, time_step, graph, time_weight, dropout, rho=0.1, readout="average", corruption="node_shuffle"):
        super(DGCN, self).__init__()
        self.time_time =time_step
        self.time_weight_weight = time_weight
        self.graph = graph
        self.num_hid = num_hid
        self.gc = GraphConvolution(device, num_feat, num_hid, time_step, time_weight)
        self.fc = nn.Linear(num_hid, num_hid, bias=False).to(device)
        self.dropout = dropout
        self.device = device
        self.prelu = nn.PReLU().to(device)
        self.num_classes = num_classes
        self.disc = Discriminator(num_hid, device)

        # 伪标签
        # lbl_rl = torch.ones(batch_size, num_nodes) #[128, 90]
        # lbl_H_fk = torch.zeros(batch_size, num_nodes)
        # lbl_X_fk = torch.zeros(batch_size, num_nodes)
        # self.lbl = torch.cat((lbl_rl, lbl_H_fk, lbl_X_fk), dim=1).to(device) #[128, 270]
        # 暂时木有用的，是否可以调用nn.Parameter使伪标签可学习
        lbl_rl = torch.ones(batch_size, 136)  # [64, 136]
        lbl_H_fk = torch.zeros(batch_size, 136)
        lbl_X_fk = torch.zeros(batch_size, 136)
        self.lbl = torch.cat((lbl_rl, lbl_H_fk, lbl_X_fk), dim=1).to(device)  # [128, 408]

        self.Adj_ = torch.nn.Parameter(torch.rand([num_nodes, num_nodes]),
                                       requires_grad=True)  # num_nodes * num_nodes维的张量; requires_grad默认值为True，表示可训练，False表示不可训练。
        self.Adj = A
        self.Pool = torch.nn.Parameter(torch.ones(size=([num_nodes])),
                                       requires_grad=True)  # torch.ones:返回一个全为1 的张量，形状由可变参数sizes定义

        self.rho = rho

        self.readout = getattr(self, "_%s" % readout)
        self.corruption = getattr(self, "_%s" % corruption)
        ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(num_feat, affine=False).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(180, 128),
            nn.Dropout(p=self.dropout),
            nn.ReLU(128),
            nn.Linear(128, 64),
            nn.ReLU(64),
            nn.Linear(64, self.num_classes)).to(device)

    def forward(self, X, A, last_embedding=0):
        X_concat = torch.cat([graph.node_features.view(1, -1, graph.node_features.shape[1]) for graph in X],0).to(self.device)
        A = F.relu(A) #[90, 90]
        B, N, D = X_concat.shape # 64, 90, 136
        X_concat = X_concat.view(-1, D)
        X_concat = self.bn0(X_concat)
        X_concat = X_concat.view(B, N, D)
        x = X_concat # torch.Size([64, 90, 136])
        x = x.view(-1, N)
        # 7. 图卷积
        HHH, weight = self.gc(x, A) # hhh: [90,5760]
        HHH = HHH.reshape(B, N, N)
        h = self.prelu(HHH) # h: [64, 90, 90]
        # print(H.shape)

        # print(self.training) # True
        # if not self.training:
        #     return h, weight

         # 池化层
        max_pool, ind = torch.max(h, dim=1)  # [64, 90]
        min_pool = torch.min(h, dim=1)[0]  # [64, 90]
        mean_pool = torch.mean(h, dim=1)  # [64, 90]
        sum_pool = torch.sum(h, dim=1)
        '''baseline: Pooling层'''
        # repeated_pool = self.Pool.repeat(h.shape[0], 1, 1) # repeat():沿着指定的维度重复tensor的数据
        # weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1) # [64, 520] torch.bmm:两个tensor的矩阵乘法； view():重新调整Tensor的形状
        # pooled = torch.cat((max_pool, weighted_pool, mean_pool), dim=1) # [64, 1560] 将tensor类型拼接起来

        '''len+mean'''
        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)  #[64, 1, 90] repeat():沿着指定的维度重复tensor的数据
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0], -1)  # [64, 520] torch.bmm:两个tensor的矩阵乘法； view():重新调整Tensor的形状
        pooled = torch.cat((weighted_pool, mean_pool), dim=1)  # [64, 180] 将tensor类型拼接起来
        # print(pooled.shape)

        x_out = self.classifier(pooled)
        return x_out, weight

        # 伪图
        # x_, neg_A = self.corruption(x, A, edge=1) # 边打乱
        # neg_X_, A_ = self.corruption(x, A, edge=0) # 节点特征打乱
        #
        #
        # neg_X = F.dropout(neg_X_, self.dropout, training=self.training)
        # neg_A = F.dropout(neg_A, self.dropout, training=self.training)
        #
        # neg_HHH, neg_weight_H = self.gc(x_, neg_A) # 边打乱
        # neg_XXX, neg_weight_X = self.gc(neg_X, A_) # 节点特征打乱
        #
        # neg_H = self.prelu(neg_HHH) # 边打乱 伪图图卷积后 [90,16384]
        # neg_X = self.prelu(neg_XXX) # 节点特征打乱 伪图图卷积后 [90,16384]
        #
        # d = 1
        # if (d==0): # 两个伪图的局部图 与 原图的全局图 的 MI
        #     g = self.readout(H)
        #     g = g.reshape(-1,self.num_hid)
        #     g = self.fc(g) # [90,90]
        #     cat_num = torch.cat((H, neg_H, neg_X)) #[270,16384]
        #     g = g.reshape(-1)
        #     x_out = torch.mv(cat_num, g) # torch.mv()：矩阵向量乘法,第一个参数是二维矩阵、第二个参数是一维向量
        #     labels = torch.cat((torch.ones(x.size(0)), torch.zeros(neg_X_.size(0)), torch.ones(x_.size(0)))).to(self.device)
        #     # print("-------labels------")
        #     # print(labels.shape) #[270]
        # elif (d==1): # 两个伪图的局部图 与 原图的全局图 的 MI (使用AAAI 2023 ST-SSL中的方法)
        #     g = self.readout(H) # 原图G的节点表示H -> 图级表示g [17408]
        #     g = torch.unsqueeze(g, dim=1)  # [17408,1]
        #     g = g.expand_as(H.transpose(0,1)).contiguous() # [17408,90]
        #     g = self.fc(g) # # [17408,90]
        #     # 鉴别器
        #     x_out = self.disc(g, H, neg_H, neg_X) #[128, 408]
        #     labels = self.lbl #[128,408]
        #     x_out = self.classifier(x_out) #[128, 6]
        #
        # return x_out, labels


    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A, edge):
        if edge: # 边的打乱
            perm_A = torch.randperm(A.size(0))  # 随机打乱后获得的数字序列
            neg_A = A[perm_A]
            A = neg_A
        else: # 节点特征的打乱
            # print("-----A-----")
            # print(X.size(0)) #128
            # print(X[torch.randperm(X.size(0))].shape)#[128,90,136]
            perm = torch.randperm(X.size(0))  # 随机打乱后获得的数字序列
            neg_X = X[perm]
            X = neg_X
            # print(X.shape) #[128,90,136]
        return X, A

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.sum()

    def _adj_corrupt(self, X, A):
        rho = self.rho
        [n, m] = A.shape
        neg_A = A.clone()
        p = np.random.rand(n, m)
        d_A = np.zeros((n, m))
        d_A[p < rho] = 1
        neg_A = np.logical_xor(neg_A.to_dense().data.cpu().numpy(), d_A)
        idx = np.nonzero(neg_A)
        d_A = torch.sparse.FloatTensor(torch.LongTensor(np.array(idx)), torch.FloatTensor(np.ones(len(idx[0]))) , \
                                       torch.Size([n, m])).cuda()
        return X, d_A

    def modularity_generator(self,G):
        """
        Function to generate a modularity matrix.
        :param G: Graph object.
        :return laps: Modularity matrix.
        """
        print("Modularity calculation.\n")
        degrees = nx.degree(G)
        e_count = len(nx.edges(G))
        modu = np.array(
            [[float(degrees[node_1] * degrees[node_2]) / (2 * e_count) for node_1 in nx.nodes(G)] for node_2 in
             tqdm(nx.nodes(G))], dtype=np.float64)
        return modu
    def get_idcatematrix(self,node_num,k,embeddings):
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings.detach().numpy())
        print("y_pred type is:{0}, shape is:{1}".format(type(y_pred), y_pred.shape))
        y_pred = torch.LongTensor(y_pred)
        ones = torch.sparse.torch.eye(k)
        y_one_hot = ones.index_select(0,y_pred)
        print("y_one_hot:{}".format(y_one_hot.size))
        return y_one_hot

    def get_exitembeddings(self, graph, embedding):
        exit_embeddings = []
        exitNode_list = sorted(list(graph.nodes()))
        for j, en in enumerate(embedding.detach().numpy()):
            if (j in exitNode_list):
                exit_embeddings.append(en)
        exit_embeddings = np.mat(exit_embeddings)
        return exit_embeddings
