import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from Utils.util import load_data_our, separate_data
from models.graphcnn_lstm_DiceNew_RML import DGCN, Graph_Inception
from Utils.pytorchtools_lstm_dice_RML import EarlyStopping

import scipy.sparse as sp
import time
import torch.nn.functional as F
from utils import accuracy, binary_accuracy, get_afldata, Comp_loss
from sklearn.metrics import f1_score, accuracy_score

criterion = nn.CrossEntropyLoss()


"""DCN-train"""
# def train(args, model, device, train_graphs, optimizer, epoch, A, time_step, features, last_embedding, patience, batch_size):
#     total_iters = args.iters_per_epoch
#     pbar = tqdm(range(total_iters), unit='batch')
#
#     loss_accum = 0
#     accum = 0
#     best_loss = float("inf")
#
#     for pos in pbar:
#         selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
#         batch_graph = [train_graphs[idx] for idx in selected_idx] # list类型求大小 128
#
#         if time_step >= 1:
#             output, labels = model(batch_graph, A, last_embedding)
#         else:
#             output, labels = model(batch_graph, A)
#         # print("-----train---------")
#         # print(output.shape)#[128， 6]
#         # print(labels.shape) #[128， 408]
#         # print(output)# 正负的值
#         # print(labels) # 0，1数组
#         labels_ = torch.LongTensor([graph.label for graph in batch_graph]).to(device) #[128] 真实数据集中的标签(RML:0~5)
#
#         loss = criterion(output, labels_)
#
#         _, pred_ = torch.max(output, 1)
#         correct = pred_.eq(labels_.view_as(pred_)).sum().cpu().item()  # view_as:该函数的作用是将调用函数的变量，转变为同参数tensor同样的形状,转换前后元素数量不变
#         acc_train = correct / float(len(batch_graph))
#         # backprop
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         loss = loss.detach().cpu().numpy()
#         loss_accum += loss
#         accum += acc_train
#
#         # report
#         pbar.set_description('epoch: %d' % (epoch))
#
#
#     average_loss = loss_accum / total_iters
#     average_acc = accum / total_iters
#     print("Training Loss : %f" % (average_loss))
#     print("Training Acc : %f" % (average_acc))
#
#
#     return average_loss
#
# def pass_data_iteratively(model, graphs, A, minibatch_size = 64):
#     model.eval()
#     output = []
#     label = []
#     idx = np.arange(len(graphs))
#     for i in range(0, len(graphs), minibatch_size):
#         sampled_idx = idx[i:i+minibatch_size]
#         if len(sampled_idx) == 0:
#             continue
#         graph = [graphs[j] for j in sampled_idx] #[[graph,]]
#         output.append(model(graph, A)[0].detach())  #  troch.detach():返回一个新的tensor,永远不需要计算其梯度
#         label.append(model(graph, A)[1].detach())
#         # print("--------------------")
#         # print(model(graph, A)[0].shape)
#     # torch.cat:除拼接维数dim数值可不同外其余维数数值需相同，方能对齐
#     return torch.cat(output, 1), torch.cat(label, 1)
# def test(args, model, features, A, test_graphs, train_graphs, device):
#     model.eval()
#     with torch.set_grad_enabled(False):
#         selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
#         batch_graph = [train_graphs[idx] for idx in selected_idx]  # list类型求大小 128
#         output, l = model(batch_graph, A)
#         _, pred_ = torch.max(output, 1)
#         labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)  # [64]
#         correct = pred_.eq(
#             labels.view_as(pred_)).sum().cpu().item()  # view_as:该函数的作用是将调用函数的变量，转变为同参数tensor同样的形状,转换前后元素数量不变
#         acc_train = correct / float(len(batch_graph))
#
#         avg_accuracy_train = round(accuracy_score(labels.cpu().detach(), pred_.cpu().detach()) * 100, 2)
#         avg_fscore_train = round(f1_score(labels.cpu().detach(), pred_.cpu().detach(), average='weighted') * 100, 2)
#
#         selected_idx = np.random.permutation(len(test_graphs))[:args.batch_size]
#         batch_graph = [test_graphs[idx] for idx in selected_idx]  # list类型求大小 128
#         output, weight = model(batch_graph, A)
#         _, pred_ = torch.max(output, 1)  # [128]
#         labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)  # [128]
#         correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()
#         acc_test = correct / float(len(batch_graph))
#
#         avg_accuracy_test = round(accuracy_score(labels.cpu().detach(), pred_.cpu().detach()) * 100, 2)
#         avg_fscore_test = round(f1_score(labels.cpu().detach(), pred_.cpu().detach(), average='weighted') * 100, 2)
#
#         print("----------------------All_ACC-------------------------------")
#         print("Accuracy train: %f test: %f" % (acc_train, acc_test))
#
#         print("----------------------avg_ACC  avg_F1-------------------------------")
#         print("Train avg_ACC: %f avg_F1: %f" % (avg_accuracy_train, avg_fscore_train))
#         print("Test avg_ACC: %f avg_F1: %f" % (avg_accuracy_test, avg_fscore_test))
#     return avg_accuracy_test, avg_fscore_test, weight
'''Inception'''
def Comp_loss(pred, label, pred_Adj, Adj, Adj_factor, Adj_factor_2, pred_Pool, Pool, Pool_factor):
    # Adj: A=(i-j)**2
    # LGC 图分类损失函数
    loss = criterion(pred, label)

    m = nn.Threshold(0, 0)
    pred_Adj = m(pred_Adj)
    # LGL 图学习损失函数
    loss += Adj_factor * (torch.mean(torch.mul(pred_Adj, Adj))) + Adj_factor_2 * (torch.sqrt(torch.mean((Adj - torch.zeros_like(Adj)) ** 2)) )
    loss += Pool_factor * torch.sqrt(torch.mean((pred_Pool - torch.zeros_like(pred_Pool)) ** 2)) # torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容
    return loss
def train(args, model, device, train_graphs, optimizer, epoch, A):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output,_ = model(batch_graph,A)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # setting loss function coefficients
        Adj_factor = 0.1
        Adj_factor_2 = 0.1
        Pool_factor = 0.0001
        loss = Comp_loss(output, labels, model.Adj.to(device), A.to(device)
                         , Adj_factor, Adj_factor_2, model.Pool.to(device), torch.ones(size=([len(batch_graph[0].g)])).to(device), Pool_factor)

        _, pred_ = torch.max(output, 1)
        correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()  # view_as:该函数的作用是将调用函数的变量，转变为同参数tensor同样的形状,转换前后元素数量不变
        acc_train = correct / float(len(batch_graph))

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        accum += acc_train
        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    average_acc = accum / total_iters
    print("loss training : %f" % (average_loss))
    # print("acc training : %f" % (average_acc))


    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, A, minibatch_size = 64):
    model.eval()
    output = []
    ind = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx], A)[0].detach())
        ind.append(model([graphs[j] for j in sampled_idx], A)[1].detach())
        weight = (model([graphs[j] for j in sampled_idx], A)[1].detach())
    # return torch.cat(output, 0), torch.cat(ind, 0)
    return torch.cat(output, 0), weight

def test(args, model, device, train_graphs, test_graphs, num_class, A):
    model.eval()

    output, weight_train = pass_data_iteratively(model, train_graphs, A)
    pred_ = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    avg_accuracy_train = round(accuracy_score(labels.cpu().detach(), pred_.cpu().detach()) * 100, 2)
    avg_fscore_train = round(f1_score(labels.cpu().detach(), pred_.cpu().detach(), average='weighted') * 100, 2)

    output, weight_test = pass_data_iteratively(model, test_graphs, A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    avg_accuracy_test = round(accuracy_score(labels.cpu().detach(), pred.cpu().detach()) * 100, 2)
    avg_fscore_test = round(f1_score(labels.cpu().detach(), pred.cpu().detach(), average='weighted') * 100, 2)

    print("Accuracy train: %f test: %f" % (acc_train, acc_test))
    print("Train avg_ACC: %f avg_F1: %f" % (avg_accuracy_train, avg_fscore_train))
    print("Test avg_ACC: %f avg_F1: %f" % (avg_accuracy_test, avg_fscore_test))

    return avg_accuracy_test, avg_fscore_test, weight_train

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Mine_Graph_RML",
                        help='name of dataset (default: Mine_Graph_RML、RAVDESS)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 5 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=2,
                        help='the index of fold in 5-fold validation. Should be less then 5.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 2)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--Normalize', type=bool, default=True, choices=[True, False],
                        help='Normalizing data')
    parser.add_argument('--patience', type=int, default=100,
                        help='Normalizing data')
    parser.add_argument('--hidden', type=int, default=272,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--rho', type=float, default=0.1,
                        help='Adj matrix corruption rate')
    parser.add_argument('--corruption', type=str, default='node_shuffle',
                        help='Corruption method')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show training process')
    parser.add_argument('--test_only', action="store_true", default=False,
                        help='Test on existing model')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load data
    graphs, num_classes, num_node = load_data_our(args.dataset, args.degree_as_tag, args.Normalize)

    time_weight_list = [0]
    embedding_list = [0]
    # 局部图的遍历
    for i in range(1):
        acc_test = 0
        f1_test = 0
        acc_test_sum_F = 0
        f1_test_sum_F = 0
        # graph_num = len(graphs)
        graph_num = 10
        for t in range(graph_num):
            print("*********************** |Graph number:|   **********************：%i" % (t))
            dice = 1.0
            num_nodes = num_node # RML数据集： num_nodes =90
            # 2.动态获取adj、特征、图
            A, features, graph = get_afldata(dice, graphs[t], num_nodes)
            A = A.to(device)  # torch.Size([90, 90])
            features = features.to(device)

            # # 距离
            A_adj = np.zeros([num_nodes, num_nodes])
            # for i in range(num_nodes):
            #     for j in range(num_nodes):
            #         A_adj[i, j] = (i - j) ** 2
            # # 手动定义
            for i in range(num_nodes-1):
                A_adj[i, i+1] = 1
                A_adj[i+1, i] = 1

            A_adj = torch.FloatTensor(A_adj).to(device)

            ##5-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
            train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

            #iniial adjacency matrix
            num_nodes = train_graphs[0].node_features.shape[0] # 90
            # print(num_nodes)



            # print("---------features.shape[1]----------")
            # print(features.shape[1]) # 90
            # print("---------train_graphs[0][0].node_features.shape[1]----------")
            # print(train_graphs[0].node_features.shape[1])  # 136
            # model = DGCN(device, num_nodes, args.batch_size, num_classes,A, num_feat=train_graphs[0].node_features.shape[1],
            #              num_hid=args.hidden,
            #              time_step=t,
            #              graph=graph,
            #              time_weight=time_weight_list[-1],
            #              dropout=args.dropout,
            #              rho=args.rho,
            #              corruption=args.corruption).to(device)
            model = Graph_Inception(args.num_layers, train_graphs[0].node_features.shape[1],
                                    num_classes, args.final_dropout,
                                    device, args.dataset, args.batch_size, num_nodes,
                                    A, t, args.hidden, A_adj, time_weight=time_weight_list[-1]).to(device)

            Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Number of Trainable Parameters= %d" % (Num_Param))
            print('Model {} : params: {:4f}M'.format(model._get_name(), Num_Param / (1024*1024)))

            # for i in range(args.fold_idx):
            # print("***********************|Fold|***********************：%i" % (i))
            train_data =  train_graphs[i]
            test_data = test_graphs[i]

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.5)

            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
            for epoch in range(1, args.epochs + 1):
                scheduler.step()
                # Train
                '''DGCN'''
                # train(args, model, device, train_graphs, optimizer, epoch, A, t, features, embedding_list[-1], args.patience, args.batch_size)
                '''Inception'''
                train(args, model, device, train_graphs, optimizer, epoch, A)
                if (epoch > 1):
                    #### Validation check
                    with torch.no_grad():
                        '''DGCN'''
                        # selected_idx = np.random.permutation(len(test_graphs))[:args.batch_size]
                        # batch_graph = [test_graphs[idx] for idx in selected_idx]  # list类型求大小 128
                        # val_out, _ = model(batch_graph, A) #[128, 6]
                        # val_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)  # [128]
                        # val_loss = criterion(val_out, val_labels)
                        # val_loss = np.average(val_loss.detach().cpu().numpy())
                        '''Inception'''
                        val_out, _ = pass_data_iteratively(model, test_graphs, A)
                        val_labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
                        val_loss = criterion(val_out, val_labels)
                        val_loss = np.average(val_loss.detach().cpu().numpy())

                    #### Check early stopping
                    early_stopping(val_loss, model)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            # Test
            model.load_state_dict(torch.load('Saved_models/checkpoint_dice_RML_2.pt'))
            '''DGCN'''
            # avg_accuracy_test, avg_fscore_test, weight = test(args, model, features, A, test_graphs, train_graphs, device)
            '''Inception'''
            avg_accuracy_test, avg_fscore_test, weight = test(args, model, device, train_graphs, test_graphs, num_classes, A)
            time_weight_list.append(weight)
            acc_test += avg_accuracy_test
            f1_test += avg_fscore_test

        # 计算所有图的结果
        acc_test_sum_F += acc_test / graph_num
        f1_test_sum_F += f1_test / graph_num
        print("                                                                               {}折交叉验证，模型ACC:{}，F1:{}".format(i, acc_test_sum_F, f1_test_sum_F))


if __name__ == '__main__':
    main()
