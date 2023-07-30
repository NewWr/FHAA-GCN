import warnings
import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.metrics import *
from utils import *
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
from models import *
import torch.nn.functional as F
from lion_pytorch import Lion


batch = 36
LR = 0.0001
Epoch = 30
weight_decay = 1e-5


warnings.filterwarnings("ignore", category=Warning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_pcc, train_type1, train_weight, train_sim, train_edge, train_softlabel, train_adj, test_pcc, test_type1, \
                                    test_weight, test_sim, test_edge, test_softlabel, test_adj = load_data1()

loss_func_ce = nn.CrossEntropyLoss()
loss_func = FocalLoss()

# MLP pre-train
model_MLP = MLP(n_feat=116, n_hid=116, nclass=2, dropout=0.5).to(device)
optimizer_mlp = optim.Adam(model_MLP.parameters(), lr=0.0001, weight_decay=1e-5)
mlp_acc = 0

for i in range(30):
        model_MLP.train()
        train_pcc, train_type1, train_weight, train_sim, train_edge = train_pcc.to(device), train_type1.to(device), \
                                train_weight.to(device), train_sim.to(device), train_edge.to(device)
        optimizer_mlp.zero_grad()
        output_mlp_w = model_MLP.forward(train_sim).to(device)
        acc_mlp, right_mlp, miss_mlp = accuracy(output_mlp_w, train_type1)
        loss_mlp = loss_func(output_mlp_w, train_type1, right_mlp, miss_mlp).to(device)
        loss_mlp.backward()
        optimizer_mlp.step()
        print('Epoch: {:04d}'.format(i + 1), 'loss_train_mlp: {:.4f}'.format(loss_mlp.item()),
              'acc_mlp: {:.4f}'.format(acc_mlp.item()))

        test_pcc, test_type1, test_weight, test_sim, test_edge = test_pcc.to(device), test_type1.to(device), \
                                test_weight.to(device), test_sim.to(device), test_edge.to(device)
        model_MLP.eval()
        output_mlp1_w = model_MLP(test_sim).to(device)
        acc_test_mlp = accuracy_mlp(output_mlp1_w, test_type1).to(device)
        loss_mlp_test = loss_func_ce(output_mlp_w, train_type1).to(device)
        print('testSet:''loss: {:.4f}'.format(loss_mlp_test.item()), 'acc: {:.4f}'.format(acc_test_mlp.item()))

        if acc_test_mlp > mlp_acc:
            mlp_acc = acc_test_mlp
            mlp_w = output_mlp_w
            mlp1_w = output_mlp1_w

print("~~~~~~~~~~~~~~~~~~MLP finish~~~~~~~~~~~~~~~~~~~~~~")


labels_for_mlp_w = one_hot_embedding(train_type1, train_type1.max().item() + 1, mlp_w).type(torch.FloatTensor)
labels_for_mlp1_w = one_hot_embedding(test_type1, test_type1.max().item() + 1, mlp1_w).type(torch.FloatTensor)

adj1_train, sadj1_train = load_graph(train_edge)
bi_adj1_train = adj1_train.matmul(adj1_train)

adj1_test, sadj1_test = load_graph(test_edge)
bi_adj1_test = adj1_test.matmul(adj1_test)

dataSet = Data.TensorDataset(train_pcc, train_type1, train_weight, train_sim, train_edge, labels_for_mlp_w, adj1_train, bi_adj1_train, train_softlabel)
dataSet1 = Data.TensorDataset(test_pcc, test_type1, test_weight, test_sim, test_edge, labels_for_mlp1_w, adj1_test, bi_adj1_test, test_softlabel)

train_loader = Data.DataLoader(dataset=dataSet, batch_size=batch, shuffle=True, drop_last=True)
test_loader = Data.DataLoader(dataset=dataSet1, batch_size=batch, shuffle=True, drop_last=True)


model = Net(116, 116, 2).to(device)
model_MLP_softlabel = MLP_Label(116, 116, 2).to(device)
optimizer = Lion(model.parameters(), lr=LR, weight_decay=weight_decay)
optimizer_mlp_softlabel = Lion(model_MLP_softlabel.parameters(), lr=0.0001, weight_decay=1e-5)

max_acc = 0
loss_list = []
acc_list = []
out_data = torch.zeros(400, 2)
real_data = torch.zeros(400, )

for epoch in range(Epoch):
    model.train()
    model_MLP_softlabel.train()
    for step, (d_p, d_t, d_w, d_s, d_e, d_mw, d_adj, d_biadj, d_sl) in enumerate(train_loader):
        d_p, d_t, d_w, d_s, d_e, d_mw, d_sl = d_p.to(device), d_t.to(device), d_w.to(device),\
            d_s.to(device), d_e.to(device), d_mw.to(device), d_sl.to(device)
        optimizer.zero_grad()
        optimizer_mlp_softlabel.zero_grad()
        output_mlp_softlabel = model_MLP_softlabel(d_p).to(device)
        output, output_fc3, prob_label = model(d_w, d_p, d_s, d_mw, d_adj, d_biadj, output_mlp_softlabel)
        acc_val, right, miss = accuracy(output, d_t)
        loss = loss_func(output, d_t, right, miss).to(device)
        # 同质性矩阵loss
        loss_lp_sum = 0
        for t in range(output_mlp_softlabel.shape[0]):
            loss_lp = torch.nn.functional.cross_entropy(prob_label[t], d_sl[t].squeeze().long())
            loss_lp_sum = loss_lp + loss_lp_sum
        loss = loss + loss_lp_sum / batch
        loss.backward()
        optimizer.step()
        optimizer_mlp_softlabel.step()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))
        if epoch % 1 == 0:
            model.eval()
            model_MLP_softlabel.eval()
            for step1, (d_p1, d_t1, d_w1, d_s1, d_e1, d_mw1, d_adj1, d_biadj1, d_sl1) in enumerate(test_loader):
                d_p1, d_t1, d_w1, d_s1, d_e1, d_mw1, d_sl1 = d_p1.to(device), d_t1.to(device), d_w1.to(device), d_s1.to(device)\
                    , d_e1.to(device), d_mw1.to(device), d_sl1.to(device)
                output_mlp_softlabel1 = model_MLP_softlabel(d_p1).to(device)
                output1, output1_fc3, prob_label1 = model(d_w1, d_p1, d_s1, d_mw1, d_adj1, d_biadj1, output_mlp_softlabel)
                # output1 = model(d_w1, d_p1, d_s1, d_mw1).to(device)
                acc_val1, right_val, miss_val = accuracy(output1, d_t1)
                loss_val1 = loss_func(output1, d_t1, right_val, miss_val).to(device)
                loss_lp_sum1 = 0
                for t in range(output_mlp_softlabel.shape[0]):
                    loss_lp = torch.nn.functional.cross_entropy(prob_label[t], d_sl[t].squeeze().long())
                    loss_lp_sum1 = loss_lp + loss_lp_sum1
                loss_val1 = loss_val1 + loss_lp_sum1 / batch
                print("Test set results:",
                      "loss= {:.4f}".format(loss_val1.item()),
                      "accuracy= {:.4f}".format(acc_val1.item()))
                loss_list.append(float(loss_val1.item()))
                acc_list.append(float(acc_val1.item()))
                if max_acc < acc_val1 and acc_val1 != 1:
                    # 各类指标
                    max_acc = acc_val1
                    TP, TN, FN, FP = stastic_indicators(output1, d_t1)
                    ACC = (TP + TN) / (TP + TN + FP + FN)
                    SEN = TP / (TP + FN)
                    SPE = TN / (FP + TN)
                    BAC = (SEN + SPE) / 2
                    Precision = TP / (TP + FP)
                    Recall = TP / (TP + FN)
                    ERate = (FP + FN) / (TP + TN + FP + FN)
                    F1 = (2 * Precision * Recall) / (Precision + Recall)
                    output2 = output1
                    real_label = d_t1
                    preds, correct_num = accuracy_output(output2, d_t1)

                    if output.shape[0] == batch:
                        out_data[step * batch:(step + 1) * output.shape[0], :] = output
                        real_data[step * batch:(step + 1) * d_t.shape[0]] = d_t


print("finish")
print(max_acc.cpu().numpy().tolist())
print('ACC:', ACC.cpu().numpy().tolist())
print('SEN:', SEN.cpu().numpy().tolist())
print('SPE:', SPE.cpu().numpy().tolist())
print('BAC:', BAC.cpu().numpy().tolist())
print('Precision:', Precision.cpu().numpy().tolist())
print('Recall:', Recall.cpu().numpy().tolist())
print('ERate:', ERate.cpu().numpy().tolist())
print('F1:', F1.cpu().numpy().tolist())
print('real_label:', real_label.cpu().numpy().tolist())
print('preds_label:', preds.cpu().numpy().tolist())
print('correct_num:', correct_num.cpu().numpy().tolist())

# 提取基因和脑区 特征提取
fc1_w = model.state_dict()['fc1.weight']
fc2_w = model.state_dict()['fc2.weight']
fc3_w = model.state_dict()['fc3.weight']
conv2_w = model.state_dict()['conv2.weight']
conv1_w = model.state_dict()['conv1.weight']

prob_label = out_data[0: batch]
real_la = real_data[0: batch]

best_fc_bad_min, best_fc_bad_max, best_fc_good_max, best_fc_good_min = fre_statis_myself(fc1_w, fc2_w, fc3_w, output1_fc3, conv1_w, conv2_w, prob_label, real_la)
gene_bad, roi_bad, gene_good, roi_good = find_roi(best_fc_bad_min, best_fc_bad_max)
gene_bad, roi_bad, gene_good, roi_good = find_roi(best_fc_good_min, best_fc_good_max)

print("finish!")
