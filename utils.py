import warnings
import pandas as pd
import scipy.sparse as sp
import torch
import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=Warning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data1():
    print("load dataset ..... ")
    train_pcc, train_type1, train_weight, train_sim, train_edge, train_softlabel, train_roiinfo, train_adj,\
    test_pcc, test_type1, test_weight, test_sim, test_edge, test_softlabel, test_roiinfo, test_adj = load_data.load()

    train_feat = torch.FloatTensor(train_pcc)
    train_type1 = onehot_encode(train_type1)
    train_type1 = torch.LongTensor(np.where(train_type1)[1])
    train_weight = torch.FloatTensor(train_weight)
    train_sim = torch.FloatTensor(train_sim)
    train_edge = torch.FloatTensor(train_edge)
    # train_softlabel = onehot_encode(train_softlabel)
    train_softlabel = torch.FloatTensor(train_softlabel)

    test_pcc = torch.FloatTensor(test_pcc)
    test_type1 = onehot_encode(test_type1)
    test_type1 = torch.LongTensor(np.where(test_type1)[1])
    test_weight = torch.FloatTensor(test_weight)
    test_sim = torch.FloatTensor(test_sim)
    test_edge = torch.FloatTensor(test_edge)
    # test_softlabel = onehot_encode(test_softlabel)
    test_softlabel = torch.FloatTensor(test_softlabel)

    # print("finish load dataset")
    return train_feat, train_type1, train_weight, train_sim, train_edge, train_softlabel, train_adj, test_pcc, test_type1, test_weight, test_sim, test_edge, test_softlabel, test_adj


def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    return labes_onehot



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), correct.tolist(), labels.shape[0] - correct.tolist()


def accuracy_mlp(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_output(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return preds, correct

def stastic_indicators(output,labels):
    TP = ((output.max(1)[1] == 1) & (labels == 1)).sum()
    TN = ((output.max(1)[1] == 0) & (labels == 0)).sum()
    FN = ((output.max(1)[1] == 0) & (labels == 1)).sum()
    FP = ((output.max(1)[1] == 1) & (labels == 0)).sum()
    return TP, TN, FN, FP


def load_graph(adj):
    adj = adj.cpu()
    sadj_sum = []
    nsadj_sum = []
    for i in range(adj.shape[0]):
        sadj = sp.coo_matrix(adj[i])
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        sadj = sadj + sp.eye(sadj.shape[0])
        nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
        nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
        sadj = torch.tensor(sadj.todense(), dtype=torch.float32)
        nsadj_sum.append(nsadj)
        sadj_sum.append(sadj)

    return torch.stack(sadj_sum), torch.stack(nsadj_sum)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def one_hot_embedding(labels, num_classes, soft):
    soft = torch.argmax(soft.exp(), dim=1)
    y = torch.eye(num_classes)
    return y[soft]


# 特征提取 重要性计算 增量搜索 最佳特征子集
def fre_statis(fc1_w, fc2_w, fc3_w, output1_fc3, p_l, e_l):
    real_label = e_l
    e_l = torch.FloatTensor(onehot_encode(e_l.tolist()))
    prob = (e_l * p_l).to(device)
    grad_fc = torch.div(prob, output1_fc3)
    grad_fc3 = torch.matmul(grad_fc, fc3_w)
    grad_fc2 = torch.matmul(grad_fc3, fc2_w)
    grad_fc1 = torch.matmul(grad_fc2, fc1_w)
    grad_fc1 = grad_fc1.reshape(len(e_l), 116, 116)

    # sort_fc, sort_index = torch.sort(grad_fc1, dim=1, descending=True)
    # sort_fc, sort_index = torch.sort(sort_fc, dim=1, descending=True)
    temp_bad = 0
    temp_good = 0
    best_fc_bad = torch.FloatTensor()
    best_fc_good = torch.FloatTensor()
    for i in range(grad_fc1.shape[0]):
        if torch.sum(grad_fc1[i]) < temp_bad and real_label[i] == 0:
            best_fc_bad = grad_fc1[i]
        if torch.sum(grad_fc1[i]) > temp_good and real_label[i] == 1:
            best_fc_good = grad_fc1[i]

    return best_fc_bad, best_fc_good


def fre_statis_myself(fc1_w, fc2_w, fc3_w, output_fc3, conv1_w, conv2_w, p_l, e_l):
    real_label = e_l
    importance3 = torch.mm(output_fc3[2], fc3_w)
    importance2 = torch.mm(importance3, fc2_w)
    importance1 = torch.mm(importance2, fc1_w)
    importance_fc = importance1.reshape(len(output_fc3[2]), 116, 116)
    importance_final = importance_fc * conv2_w
    importance_final = importance_final * conv1_w
    importance_final = F.relu(importance_final)
    temp_bad_min = 0
    temp_bad_max = 0
    temp_good_min = 0
    temp_good_max = 0
    best_fc_bad_min = torch.FloatTensor()
    best_fc_bad_max = torch.FloatTensor()
    best_fc_good_min = torch.FloatTensor()
    best_fc_good_max = torch.FloatTensor()
    # for i in range(importance1.shape[0]):
    #     if torch.sum(importance1[i]) > temp_bad and real_label[i] == 0:
    #         best_fc_bad = importance1[i]
    #     if torch.sum(importance1[i]) > temp_good and real_label[i] == 1:
    #         best_fc_good = importance1[i]

    for i in range(importance_final.shape[0]):
        if torch.sum(importance_final[i]) < temp_bad_min and real_label[i] == 0:
            best_fc_bad_min = importance_final[i]
        if torch.sum(importance_final[i]) > temp_bad_max and real_label[i] == 0:
            best_fc_bad_max = importance_final[i]
        if torch.sum(importance_final[i]) > temp_good_max and real_label[i] == 1:
            best_fc_good_max = importance_final[i]
        if torch.sum(importance_final[i]) < temp_good_min and real_label[i] == 1:
            best_fc_good_min = importance_final[i]

    return best_fc_bad_min, best_fc_bad_max, best_fc_good_max, best_fc_good_min


def find_roi(fc_bad, fc_good):
    gene = pd.read_excel("D:\\DAIMA\\Py38\\Code\\HOS-GCN\\gene.xlsx")
    roi = pd.read_excel("D:\\DAIMA\\Py38\\Code\\HOS-GCN\\aal.xlsx")
    fc_bad = pd.DataFrame((F.relu(fc_bad).cpu()).detach().numpy())
    gene_sum_bad = fc_bad.apply(lambda x: x.sum(), axis=1)
    roi_sum_bad = fc_bad.apply(lambda x: x.sum(), axis=0)
    gene_bad = gene
    roi_bad = roi
    gene_bad['scores'] = gene_sum_bad
    roi_bad['scores'] = roi_sum_bad
    gene_bad = gene_bad.sort_values(by="scores", ascending=False, axis=0)
    roi_bad = roi_bad.sort_values(by="scores", ascending=False, axis=0)
    fc_good = pd.DataFrame((F.relu(fc_good).cpu()).detach().numpy())
    gene_sum_good = fc_good.apply(lambda x: x.sum(), axis=1)
    roi_sum_good = fc_good.apply(lambda x: x.sum(), axis=0)
    gene_good = gene
    roi_good = roi
    gene_good['scores'] = gene_sum_good
    roi_good['scores'] = roi_sum_good
    gene_good = gene_good.sort_values(by="scores", ascending=False, axis=0)
    roi_good = roi_good.sort_values(by="scores", ascending=False, axis=0)

    roi_gene = pd.DataFrame()
    for i in range(116):
        for j in range(116):
            roi_gene.loc[i, j] = str(roi.iloc[i, 3]) + ' ' + str(gene.iloc[j, 1])
    return gene_bad, roi_bad, gene_good, roi_good


if __name__ == '__main__':
    train_pcc, train_type1, train_weight, train_sim, train_edge, \
        test_pcc, test_type1, test_weight, test_sim, test_edge = load_data()