import os
from utils import *
import pandas as pd
from torch_geometric.data import Data
import warnings
from utils import *
import numpy as np
from itertools import chain
import math
dirPath = 'D:\DAIMA\Py38\Code\MatFiles\Compare\Hyper\w0.3_20'

# 数据集读取

Pearson = []
Type = []
Weight = []
Sim = []
Edge = []
SoftLabel = []
RoiInfo = []
Adj = []

fileName = os.listdir(dirPath)
for file in fileName:
    filePath = dirPath + '/' + file
    # print(filePath)
    data = np.load(filePath, allow_pickle=True)
    pearson = data.item().get('Feature')
    pearson[np.isnan(pearson)] = np.nanmean(pearson)

    type = data.item().get('Type')

    weight = data.item().get('Weight')
    weight[np.isnan(pearson)] = np.nanmean(weight)

    sim = data.item().get('Sim')
    sim[np.isnan(pearson)] = np.nanmean(sim)

    edge = data.item().get('HOGfeature')
    edge[np.isnan(pearson)] = np.nanmean(edge)

    softlabel = data.item().get('SoftLabel')
    roiinfo = data.item().get('RoiInfo')

    adj = np.array([0, 0])

    Pearson.append(pearson)
    Type.append(type)
    Weight.append(weight)
    Sim.append(sim)
    Edge.append(np.array(edge))
    SoftLabel.append(softlabel)
    RoiInfo.append(roiinfo)
    Adj.append(adj)

label = np.concatenate((np.zeros(272), np.ones(226)))
np.random.shuffle(shuffle_idx)

pcc = np.array(Pearson)[shuffle_idx]
type1 = np.array(label)[shuffle_idx]
weight = np.array(Weight)[shuffle_idx]
sim = np.array(Sim)[shuffle_idx]
edge = np.array(Edge)[shuffle_idx]
softlabel = np.array(SoftLabel)[shuffle_idx]
roiinfo = np.array(RoiInfo)[shuffle_idx]
adj = np.array(Adj)[shuffle_idx]

print("finish data read")

train_id = range(0, 426)
test_id = range(426, 498)



def load():
    return pcc[train_id], type1[train_id], weight[train_id], sim[train_id], edge[train_id], softlabel[train_id], adj[train_id], \
           roiinfo[train_id], pcc[test_id], type1[test_id], weight[test_id], sim[test_id], edge[test_id], softlabel[test_id], roiinfo[test_id], adj[test_id]