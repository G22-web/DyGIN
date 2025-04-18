import os
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import time
def Comp_loss(pred, label, pred_Adj, Adj, Adj_factor, pred_Pool, Pool, Pool_factor):
    # Adj: A=(i-j)**2
    loss = criterion(pred, label)
    m = nn.Threshold(0, 0)
    pred_Adj = m(pred_Adj)
    loss += Adj_factor * (torch.mean(torch.mul(pred_Adj, Adj)) + torch.sqrt(torch.mean((Adj - torch.zeros_like(Adj)) ** 2)) )
    loss += Pool_factor * torch.sqrt(torch.mean((pred_Pool - torch.zeros_like(pred_Pool)) ** 2))

    return loss
def getSimilariy_modified(OneZeromatrix,node_num,graphs):
    # similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    similar_matrix = torch.zeros(size=(node_num, node_num),dtype=float)
    graph = graphs
    for g in graph:
        edges_list = list(g.g.edges())
        node_list = g.node_tags
    for i, node in enumerate(node_list):
        for g in graph:
            # g.neighbors = [[] for i in range(len(g.g))]
            neibor_i_list = g.neighbors[node]
            first_neighbor = neibor_i_list
            for k, second_nighbor in enumerate(first_neighbor):
                second_list = g.neighbors[second_nighbor]
                neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
            neibor_i_num = len(first_neighbor)
            for j, node_j in enumerate(neibor_i_list):
                neibor_j_list = list(g.neighbors[node_j])
                neibor_j_num = len(neibor_j_list)
                commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]
                commonNeighbor_num = len(commonNeighbor_list)
                neibor_i_num_x = neibor_i_num
                if (i,j) in edges_list:
                    commonNeighbor_num = commonNeighbor_num + 2
                    neibor_j_num = neibor_j_num + 1
                    neibor_i_num_x = neibor_i_num + 1
                similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x)
    return similar_matrix, graph
def getSimilariy_modified_our(OneZeromatrix,node_num,graphs):
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    graph = graphs
    edges_list = list(graph.g.edges())
    node_list = graph.node_tags

    for i, node in enumerate(node_list): 
        # for g in graph:
        neibor_i_list = graph.neighbors[node] 
        first_neighbor_ = neibor_i_list
        for k, first_neighbor in enumerate(first_neighbor_):
            second_list = graph.neighbors[first_neighbor]
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
        neibor_i_num = len(first_neighbor_)
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors[node_j])
            neibor_j_num = len(neibor_j_list)
            commonNeighbor_list = [x for x in first_neighbor_ if x in neibor_j_list]
            commonNeighbor_num = len(commonNeighbor_list)
            neibor_i_num_x = neibor_i_num
            if (i, j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
            neibor_j_num = neibor_j_num + 1
            neibor_i_num_x = neibor_i_num + 1
            if (node == node_j):
                similar_matrix[node, node_j] = 1
            else:
                similar_matrix[node, node_j] = (commonNeighbor_num + neibor_j_num / (
                        neibor_i_num_x + neibor_j_num)) / (neibor_i_num_x + 1)  # Dice_new
    return similar_matrix, graph
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def get_afldata(dice, graphs, num_nodes):
    adj = sp.lil_matrix((num_nodes, num_nodes), dtype=int)
    # for g in graphs:
    edges = [list(pair) for pair in graphs.g.edges()]  
    for list_edg in edges:
        from_id, to_id = list_edg[0], list_edg[1]
        if from_id == to_id:
            continue
        adj[int(from_id), int(to_id)] = 1
        adj[int(to_id), int(from_id)] = 1
    features_adj = sp.coo_matrix(adj,dtype=float)

    t1 = time.time()
    similairity_matirx, graph = getSimilariy_modified_our(adj, num_nodes, graphs)

    adj = adj + dice * similairity_matirx 
    features_adj = normalize(features_adj)
    features_adj = torch.FloatTensor(np.array(features_adj.todense()))
    
    adj = normalize((adj + np.eye(adj.shape[0]))) 
    adj = torch.FloatTensor(adj) 
    return adj, features_adj, graph
def get_degree_feature_list(edges_list_path, node_num, init='one-hot'):
    x_list = []
    max_degree = 0
    adj_list = []
    degree_list = []
    ret_degree_list = []
    file_num = len(os.listdir(edges_list_path))
    edges_dir_list = []
    for i in range(file_num):
        f_name = "edges_t" + str(i + 1) + ".txt"
        edges_dir_list.append(f_name)
    for i in range(len(edges_dir_list)):
        edges_path = os.path.join(edges_list_path, edges_dir_list[i])
        adj_lilmatrix = get_adj_lilmatrix(edges_path,node_num)
        # node_num = len(adj)
        adj = sp.coo_matrix(adj_lilmatrix)
        adj_list.append(adj)
        degrees = adj.sum(axis=1).astype(np.int)
        max_degree = max(max_degree, degrees.max())
        degree_list.append(degrees)
        ret_degree_list.append(torch.FloatTensor(degrees).cuda() if torch.cuda.is_available() else degrees)
    for i, degrees in enumerate(degree_list):
        if init == 'gaussian':
            fea_list = []
            for degree in degrees:
                fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
            fea_arr = np.array(fea_list)
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
            return x_list, fea_arr.shape[1], ret_degree_list
        elif init == 'combine':
            fea_list = []
            for degree in degrees:
                fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
            fea_arr = np.array(fea_list)
            ###################
            fea_arr = np.hstack((fea_arr, adj_list[i].toarray()))
            ###################
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
            return x_list, fea_arr.shape[1], ret_degree_list
        elif init == 'one-hot':  # one-hot degree feature
            degrees = np.asarray(degrees,dtype=int).flatten()
            one_hot_feature = np.eye(max_degree + 1)[degrees]
            x_list.append(one_hot_feature.cuda() if torch.cuda.is_available() else one_hot_feature)

        else:
            raise AttributeError('Unsupported feature initialization type!')
    return x_list, max_degree + 1


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # print("labels_onehot".format(labels_onehot))
    return labels_onehot

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)

def get_vttdata(node_num):
    all=range(node_num)
    idxes = np.random.choice(all, 20 + 44 + 147)
    idx_train, idx_val, idx_test = idxes[:20], idxes[20:-147], idxes[-147:]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    _, pred_ = torch.max(outputs, 1)  # [751]
    print(pred_.shape)
    preds = preds.max(1, keepdim=True)[1].type_as(labels) 
    print(preds.shape)

    correct = preds.eq(labels).double() 
    correct = correct.sum() 
    return correct / len(labels) 

def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def check_and_creat_dir(file_url):
    file_gang_list = file_url.split('/')
    if len(file_gang_list) > 1:
        [fname, fename] = os.path.split(file_url)
        # print(fname, fename)
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            return None

    else:
        return None


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)
