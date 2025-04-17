import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import math
from torch.optim.optimizer import Optimizer, required

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

"""
 Mine_Graph_RML:
    # classes: 6
    # maximum node tag: 90
    # data: 1503

"""
def load_data(dataset, degree_as_tag, Normalize):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}


    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            # print(f.readline().strip())
            # print(row)
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph() # 定义一个无向图
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                # print(row)
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    g.add_node(j, att=attr)
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                # if tmp > len(row):
                node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            if not (dataset=="Mine_Graph")or not (dataset == "Mine_Graph_test"):
                assert len(g) == n
            # print("-----------node-------------")
            # print(node_tags) # list整数列表
            # print("------------edge------------------")
            # print(n_edges) # int类型，RML:90
            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   提取唯一标签
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(g.g.nodes[0]['att']))
        for i in range(len(g.node_tags)-2):
            if (Normalize):
                g.node_features[i]=torch.FloatTensor(g.g.nodes[i]['att']/g.g.nodes[i]['att'].max())
            else:
                g.node_features[i] = torch.FloatTensor(g.g.nodes[i]['att'])



    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def load_data_our(dataset, degree_as_tag, Normalize):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    sg_list = []
    g_list = []
    label_dict = {}
    feat_dict = {}
    num_node= 0
    t = 0

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) # 1503
        for i in range(n_g):
            row = f.readline().strip().split() # f.readline():90 0； strip():去除空格； row:形式如下：['90', '1']到，['90', '5']第二维不固定
            n, l = [int(w) for w in row] # n:row的第1维  l:row的第2维
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
            g = nx.Graph() # 定义一个无向图，Graph with 0 nodes and 0 edges
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j) # 添加节点
                # print("--------添加节点后的图g-------")
                # print(g)
                # print(g.number_of_nodes()) #查看g中有多少个结点 ,依次+1上升
                row = f.readline().strip().split()
                # print("--------row-------")
                # print(row) # 从txt文件的第三行开始遍历

                tmp = int(row[1]) + 2
                # print("------------tmp-----------")
                # print(tmp)
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else: # 执行
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    # print("--------row-----------")
                    # print(row)
                    # print("--------attr-----------")
                    # print(attr)
                    g.add_node(j, att=attr)
                    # print("--------添加节点后的图g-----------")
                    # print(g)
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                # if tmp > len(row):
                node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])# 添加边
                # print("--------添加边后的图g-------")
                # print(g)

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # if not (dataset=="Mine_Graph")or not (dataset == "Mine_Graph_test"):
            #     assert len(g) == n
            # print("-----------node-------------")
            # print(node_tags) # 一维矩阵：整数
            # print("------------edge------------------")
            # print(n_edges) # int，RML:90
            g_list.append(S2VGraph(g, l, node_tags)) # len(g_list):1503个局部图 每个图有90个节点，89条边
            num_node = n

    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))] #len(g.g):2；
        for i, j in g.g.edges(): #i = 89; j = 88
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        # print("-----------max_neighbor-------------")
        # print( g.max_neighbor ) # 2

        g.label = label_dict[g.label]
        # print("-----------g.label-------------")
        # print(g.label)  # 0到5，代表RML的6中情感

        edges = [list(pair) for pair in g.g.edges()]
        # print("-----------edges-------------")
        # print(edges)  # [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 89]]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   提取唯一标签
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(g.g.nodes[0]['att']))
        for i in range(len(g.node_tags)-2):
            if (Normalize):
                g.node_features[i]=torch.FloatTensor(g.g.nodes[i]['att']/g.g.nodes[i]['att'].max())
            else:
                g.node_features[i] = torch.FloatTensor(g.g.nodes[i]['att'])



    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))
    # print("--------g_list.len-----------")
    # print(len(g_list))# 1503

    return g_list, len(label_dict), num_node

def separate_data_our(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=fold_idx, shuffle = True, random_state = seed) # shuffle:如果设置为True，则会先打乱顺序再做划分; random_state:只有当shuffle设置为True的时候才会生效。当设定某个值时，模型的训练集和测试集就固定了，方便复现结果

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    # 原始的
    # train_idx, test_idx = idx_list[fold_idx]
    #
    # train_graph_list = [graph_list[i] for i in train_idx]
    #
    # test_graph_list = [graph_list[i] for i in test_idx]
    #
    # return train_graph_list, test_graph_list
    train_portions = []
    test_portions = []
    for j in range(fold_idx):
        train_idx, test_idx = idx_list[j]

        train_portions.append([graph_list[i] for i in train_idx])
        test_portions.append([graph_list[i] for i in test_idx])

    return train_portions, test_portions

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


