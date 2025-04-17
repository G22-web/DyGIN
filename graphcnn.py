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
from layers import GraphConvolution



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
        repeated_A = A.repeat(b, 1, 1)  # #torch.Size([64, 90, 90])

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
        self.GCN_1 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 128) / 4), 128)
        self.GCN_2 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 64) / 4), 64)
    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1))])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def forward(self, A, h, padded_neighbor_list, t):
        b, c, d = h.shape
        out_1 = self.maxpool(h.view(-1, d), padded_neighbor_list).view(h.shape)
        out_2 = self.GCN_1(h, A, t)
        out_3 = self.GCN_2(h, A, t)
        out = torch.cat((out_1, out_2, out_3), dim=2)

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
        for i in range(self.num_layers):
            self.Inceptions.append(Inception_layer(input_dim+c))
            c += 192

        cell_args = u.Namespace({})
        cell_args.rows = input_dim  # 136
        cell_args.cols = input_dim

        self.evolve_weights = mat_LSTM_cell(cell_args, device)

        self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, input_dim), requires_grad=True)

        self.Linear = nn.Linear(136, 136).to(device)
        self.gru = nn.GRUCell(136, 136).to(device)  
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
            self.time_weight = self.time_weight.reshape(1, 136, 136)
            self.GCN_weights = self.transformer_encoder(self.time_weight)
            self.GCN_weights = self.GCN_weights.reshape(136, 136)

            self.dgcn_weight = self.GCN_weights
            self.dgcn_weight = torch.nn.Parameter(self.dgcn_weight,  requires_grad=True) 



            ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(input_dim, affine=False)
        

        #Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
            nn.Linear(1040, 512),
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

        B, N, D = X_concat.shape # 64, 90, 136

        X_concat = X_concat.view(-1, D)
        X_concat = self.bn0(X_concat)
        X_concat = X_concat.view(B, N, D)


        padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)


        h = X_concat #[64, 90, 136]

        if self.time_step >= 1:
            h = h.reshape(-1, 136) 
            self.first_weight = self.dgcn_weight
          

            h = torch.mm(h, self.first_weight)
            h = h.view(B, N, D)
            for layer in self.Inceptions:
                h = layer(A, h, padded_neighbor_list,self.time_step)
        else:
            h = h.reshape(-1, 136)
            self.first_weight = self.weight
            h = torch.mm(h, self.first_weight) 
            h = h.view(B, N, D) #[64, 90, 136]

            for layer in self.Inceptions:
                h = layer(A, h, padded_neighbor_list, self.time_step)

        max_pool,ind = torch.max(h,dim=1) #[64, 520]
        min_pool = torch.min(h,dim=1)[0] #[64, 520]
        mean_pool = torch.mean(h,dim=1) #[64, 520]
        sum_pool= torch.sum(h, dim=1)

        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1) 
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0], -1) 
        pooled = torch.cat((weighted_pool, mean_pool), dim=1)  
        score = self.classifier(pooled)
        return score, self.first_weight

    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A, edge):
        if edge:  
            perm_A = torch.randperm(A.size(0)) 
            neg_A = A[perm_A]
            A = neg_A
        else:
            perm = torch.randperm(X.size(0))  
            neg_X = X[perm]
            X = neg_X
        return X, A

class mat_LSTM_cell_2(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args  ##arg.rows = in_feats; arg.cols= out_feats
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
        z_topk = prev_Z  
        prev_Q = prev_Q.to(device) 

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)


        h_cap = reset * prev_Q 
        h_cap = self.htilda(z_topk, h_cap) 

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_LSTM_gate_2(torch.nn.Module):
    def __init__(self, rows, cols, activation, device):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device) 
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

        x = x.reshape(136, -1)
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)
        return out


class mat_LSTM_cell(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args  ##arg.rows = in_feats; arg.cols= out_feats
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
        prev_Q = prev_Q.to(device)  
        prev_Q = prev_Q
        z_topk = prev_Q 

        update = self.update(z_topk, prev_Q) 
        reset = self.reset(z_topk, prev_Q)  

        h_cap = reset * prev_Q  
        h_cap = h_cap.transpose(0, 1)
        h_cap = self.htilda(z_topk, h_cap) 

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_LSTM_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation, device):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device)
        self.reset_param(self.W)

        self.U = torch.nn.Parameter(torch.Tensor(rows, rows)).to(device)
        self.reset_param(self.U)

        self.bias = torch.nn.Parameter(torch.zeros(rows, cols)).to(device)

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

        h = F.relu(self.GCN_1(X_concat, A))
        h = F.softmax(self.GCN_2(h, A))


        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1)
        pooled = weighted_pool

        score = self.classifier(pooled)

        return score

class Discriminator(nn.Module):
    def __init__(self, n_h, device):
        super(Discriminator, self).__init__()

        self.net = nn.Bilinear(n_h, n_h, 1).to(device) 

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

        lbl_rl = torch.ones(batch_size, 136)  # [64, 136]
        lbl_H_fk = torch.zeros(batch_size, 136)
        lbl_X_fk = torch.zeros(batch_size, 136)
        self.lbl = torch.cat((lbl_rl, lbl_H_fk, lbl_X_fk), dim=1).to(device)  

        self.Adj_ = torch.nn.Parameter(torch.rand([num_nodes, num_nodes]),
                                       requires_grad=True)  
        self.Adj = A
        self.Pool = torch.nn.Parameter(torch.ones(size=([num_nodes])),
                                       requires_grad=True) 

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
        HHH, weight = self.gc(x, A) # hhh: [90,5760]
        HHH = HHH.reshape(B, N, N)
        h = self.prelu(HHH) # h: [64, 90, 90]

        max_pool, ind = torch.max(h, dim=1)  # [64, 90]
        min_pool = torch.min(h, dim=1)[0]  # [64, 90]
        mean_pool = torch.mean(h, dim=1)  # [64, 90]
        sum_pool = torch.sum(h, dim=1)

        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1) 
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0], -1)  
        pooled = torch.cat((weighted_pool, mean_pool), dim=1) 

        x_out = self.classifier(pooled)
        return x_out, weight


    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A, edge):
        if edge: 
            perm_A = torch.randperm(A.size(0))  
            neg_A = A[perm_A]
            A = neg_A
        else: 
            perm = torch.randperm(X.size(0)) 
            neg_X = X[perm]
            X = neg_X
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
