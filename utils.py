import os
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import time
def Comp_loss(pred, label, pred_Adj, Adj, Adj_factor, pred_Pool, Pool, Pool_factor):
    # Adj: A=(i-j)**2
    # LGC 图分类损失函数
    loss = criterion(pred, label)
    m = nn.Threshold(0, 0)
    pred_Adj = m(pred_Adj)
    # LGL 图学习损失函数
    loss += Adj_factor * (torch.mean(torch.mul(pred_Adj, Adj)) + torch.sqrt(torch.mean((Adj - torch.zeros_like(Adj)) ** 2)) )
    loss += Pool_factor * torch.sqrt(torch.mean((pred_Pool - torch.zeros_like(pred_Pool)) ** 2)) # torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容

    return loss
def getSimilariy_modified(OneZeromatrix,node_num,graphs):
    # similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    similar_matrix = torch.zeros(size=(node_num, node_num),dtype=float)
    graph = graphs
    # print("-----------graph-----------")
    # print(graph)
    for g in graph:
        edges_list = list(g.g.edges())
        # print("-----------edges_list-----------")
        # print(edges_list)# [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89)]

        node_list = g.node_tags
        # print("-----------node_list-----------")
        # print(node_list)# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]

    for i, node in enumerate(node_list):
        # print("-----------node-----------")
        # print(node) # 0
        # graph.neighbors(node): 返回节点 node 的所有邻居的迭代器
        for g in graph:
            # g.neighbors = [[] for i in range(len(g.g))]
            neibor_i_list = g.neighbors[node]
            # print("-----------neibor_i_list-----------")
            # print(neibor_i_list) # [1]
            first_neighbor = neibor_i_list
            for k, second_nighbor in enumerate(first_neighbor):
                # print("-----------second_nighbor-----------")
                # print(second_nighbor) # 1
                second_list = g.neighbors[second_nighbor]
                # print("-----------second_list-----------")
                # print(second_list) # [0,2]
                neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
                # print("-----------neibor_i_list-----------")
                # print(neibor_i_list) # [0,1,2]
            neibor_i_num = len(first_neighbor)
            # print("-----------neibor_i_num-----------")
            # print(neibor_i_num) # 1
            for j, node_j in enumerate(neibor_i_list):
                # print("-----------node_j-----------")
                # print(node_j) # 0
                neibor_j_list = list(g.neighbors[node_j])
                # print("-----------neibor_j_list-----------")
                # print(neibor_j_list)  # [1]
                neibor_j_num = len(neibor_j_list)
                # print("-----------neibor_j_num-----------")
                # print(neibor_j_num)  # 1
                commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]
                # print("-----------commonNeighbor_list-----------")
                # print(commonNeighbor_list)  #[1]
                commonNeighbor_num = len(commonNeighbor_list)
                # print("-----------commonNeighbor_num-----------")
                # print(commonNeighbor_num)  #1
                neibor_i_num_x = neibor_i_num
                # print("---------i,j-------")
                # print(i,j)
                if (i,j) in edges_list:
                    commonNeighbor_num = commonNeighbor_num + 2
                    # print("-----------commonNeighbor_num-----------")
                    # print(commonNeighbor_num)  #

                    neibor_j_num = neibor_j_num + 1
                    # print("-----------neibor_j_numzz-----------")
                    # print(neibor_j_num)  #

                    neibor_i_num_x = neibor_i_num + 1
                    # print("-----------neibor_i_num_x-----------")
                    # print(neibor_i_num_x)  #
                similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x) # 对应论文中的公式（2）Dice_new
                # print("-----------node-----------")
                # print(node)  #
                # print("-----------node_j-----------")
                # print(node_j)  #
                # print("-----------similar_matrix[node, node_j]-----------")
                # print(similar_matrix[node, node_j])  #
    return similar_matrix, graph
def getSimilariy_modified_our(OneZeromatrix,node_num,graphs):
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    # similar_matrix = torch.zeros(size=(node_num, node_num),dtype=float)
    graph = graphs
    # print("-----------graph-----------")
    # print(graph)
    # for g in graph:
    edges_list = list(graph.g.edges())
    # print("-----------edges_list-----------")
    # print(edges_list)# [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89)]

    node_list = graph.node_tags
    # print("-----------node_list-----------")
    # print(node_list)# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]

    for i, node in enumerate(node_list): # enumerate():既要遍历索引又要遍历元素
        # for g in graph:
        neibor_i_list = graph.neighbors[node]  # 返回节点 node 的所有邻居的迭代器
        first_neighbor_ = neibor_i_list
        for k, first_neighbor in enumerate(first_neighbor_):
            # print("-----------first_neighbor-----------")
            # print(first_neighbor) # 1
            second_list = graph.neighbors[first_neighbor]
            # print("-----------second_list-----------")
            # print(second_list) # [0,2]
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
            # print("-----------neibor_i_list-----------")
            # print(neibor_i_list) # [0,1,2]
        neibor_i_num = len(first_neighbor_)
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors[node_j])
            neibor_j_num = len(neibor_j_list)
            commonNeighbor_list = [x for x in first_neighbor_ if x in neibor_j_list]
            commonNeighbor_num = len(commonNeighbor_list)
            neibor_i_num_x = neibor_i_num
            # print("---------i,j-------")
            # print(i,j)
            if (i, j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
            neibor_j_num = neibor_j_num + 1
            neibor_i_num_x = neibor_i_num + 1
            # print("-----------commonNeighbor_num-----------")
            # print(commonNeighbor_num)  # 1
            if (node == node_j):
                similar_matrix[node, node_j] = 1
            else:
                similar_matrix[node, node_j] = (commonNeighbor_num + neibor_j_num / (
                        neibor_i_num_x + neibor_j_num)) / (neibor_i_num_x + 1)  # 对应论文中的公式（2）Dice_new
            # print("-----------node-----------")
            # print(node)  #
            # print("-----------node_j-----------")
            # print(node_j)  #
            # print("-----------similar_matrix[node, node_j]-----------")
            # print(similar_matrix[node, node_j])  #
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
    ###Adj matrix, 随机
    # adj = torch.nn.Parameter(torch.rand([num_nodes, num_nodes]), requires_grad=True) #随机
    # print("---------adj--------------")
    # print(adj)
    # features_adj = sp.coo_matrix(adj.detach.numpy, dtype=float) #稀疏矩阵,得到一个有向图

    # # 初始化为给定维度的全零数组,根据距离定义Adj,容易形成次优图
    # adj = np.zeros([num_nodes, num_nodes])
    # or
    # adj = sp.lil_matrix((num_nodes, num_nodes), dtype=int) # 增量构造稀疏矩阵
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         adj[i, j] = (i - j) ** 2
    # print("---------adj1--------------")
    # print(adj)
    adj = sp.lil_matrix((num_nodes, num_nodes), dtype=int) # 增量构造稀疏矩阵
    # for g in graphs:
    edges = [list(pair) for pair in graphs.g.edges()]  # 单向
    for list_edg in edges:
        from_id, to_id = list_edg[0], list_edg[1]
        if from_id == to_id:
            continue
        adj[int(from_id), int(to_id)] = 1
        adj[int(to_id), int(from_id)] = 1
    # print("---------adj--------------")
    # print(adj)

    # 稀疏矩阵,得到一个有向图
    features_adj = sp.coo_matrix(adj,dtype=float)
    # print("---------features_adj-----------")
    # # print(features_adj.shape[1]) # RML: 90
    # print(features_adj)  # RML: 90

    t1 = time.time()
    # 求aij的邻居的节点相似度
    similairity_matirx, graph = getSimilariy_modified_our(adj, num_nodes, graphs)
    # print("-------similairity_matirx----------")
    # print(similairity_matirx)

    adj = adj + dice * similairity_matirx # 对应论文中的图4（b）聚合策略，矩阵不对称 （0，1）与（1，0）的值不一样
    # print("-------adj-Dice：----------")
    # print(adj)
    features_adj = normalize(features_adj) # 对特征做了归一化的操作
    features_adj = torch.FloatTensor(np.array(features_adj.todense()))#将numpy的数据转换成torch格式
    '''
        论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
        和下面两个语句是等价的，仅仅是为了产生对称的矩阵 
        adj_2 = adj + adj.T.multiply(adj.T > adj)
        adj_3 = adj + adj.T
    '''
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 非对称邻接矩阵转变为对称邻接矩阵(有向图转无向图) 的固定操作
    # print("------adj有向图转无向图后：---------")
    # print(adj)

    # 当adj随机初始化时：
    # adj = normalize((adj + torch.eye(adj.shape[0])).detach().numpy()) # 对A+I归一化 等于 对应公式A~=A+IN
    # adj = torch.tensor(adj)
    adj = normalize((adj + np.eye(adj.shape[0])))  # 对A+I归一化 等于 对应公式A~=A+IN
    # print("------adj，A+I：---------")
    # print(adj)
    adj = torch.FloatTensor(adj) # 邻接矩阵转为tensor处理
    # print("------adj，邻接矩阵转为tensor处理：---------")
    # print(adj)
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
    print("--------------")
    preds = outputs.gt(0).type_as(labels)# 把outputs里面值与0比較,大于0设为1,否则设为0
    _, pred_ = torch.max(outputs, 1)  # [751]
    print(pred_.shape)
    preds = preds.max(1, keepdim=True)[1].type_as(labels)  # output.max(1)按行来寻找最大值的索引位置，返回该最大值[0]和它的索引位[1]
    print(preds.shape)

    correct = preds.eq(labels).double() # preds.eq(labels).double()返回01，便于统计总命中数量
    correct = correct.sum() # 求出预测正确的标签数量
    return correct / len(labels) # 预测正确的数量除以总数

def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels) # 把outputs里面值与0比較,大于0设为1,否则设为0
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
