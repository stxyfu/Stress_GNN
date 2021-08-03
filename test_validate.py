# Draw prediction results for validation

import torch
from torch_geometric.data import DataLoader
import time, datetime
import random, os
import numpy as np
import models, utils, train
from utils import print
import time
import os

if __name__=='__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.cuda.empty_cache() # clear gpu
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # test conventional graph embedding method, files with "_bc2.gzip" appendix
    # Directory of the file using the conventional method
    # testPath = "C:/Data2_Conv_BC2_5000\\0_1_5000\\0_1_1538_gnn_bc2.gzip"
    # # 3-layer GCN
    # net = models.StressGCN_Conv(18, 64, 3)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_3layerGCN"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    # # 8-layer GCN
    # net = models.StressGCN_Conv(18, 256, 8)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_8layerGCN"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    # # 3-layer DeepGCN
    # net = models.StressDeepGCN(18, 64, 1, 3)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_3layerDeepGCN"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    # # 3-layer DeepGCN
    # net = models.StressDeepGCN(18, 256, 1, 8)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_8layerDeepGCN"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    # # GAT
    # net = models.StressGCN_GAT_8layer(18, 64)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_GAT"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    # gUNet
    # net = models.StressGCN_UNet(18, 32, 64)
    # net.load_state_dict(torch.load("./results/stressGCN_convention_gUnet"))
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath))

    # test BOGE method with stress field, files with "_high.gzip" appendix
    net = models.StressDeepGCN(18, 64, 1, 3)
    net.load_state_dict(torch.load("./results/stress_20000"))
    testPath = "C:/Data2_Conv_20000_High\\0_1_20000\\0_1_14575_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\0_2_20000\\0_2_5564_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\0_3_20000\\0_3_7301_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\1_1_20000\\1_1_11437_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\1_2_20000\\1_2_16791_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\1_3_20000\\1_3_16618_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\2_1_20000\\2_1_8147_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\2_2_20000\\2_2_14251_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))
    testPath = "C:/Data2_Conv_20000_High\\2_3_20000\\2_3_15992_fullBC_high.gzip"
    train.testDrawGen(net, testPath, device, os.path.basename(testPath))

    # test BOGE method with stress field, files with "_opt_gnn.gzip" appendix
    # net = models.StressDeepGCN(18, 64, 1, 3)
    # net.load_state_dict(torch.load("./results/topo_5000"))
    # testPath = "C:/GNN_Dataset_Topo\\0_1\\0_1_1538_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\0_2\\0_2_1321_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\0_3\\0_3_1112_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\1_1\\1_1_3180_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\1_2\\1_2_3928_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\1_3\\1_3_3917_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\2_1\\2_1_2476_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\2_2\\2_2_2226_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")
    # testPath = "C:/GNN_Dataset_Topo\\2_3\\2_3_1855_opt_gnn.gzip"
    # train.testDrawGen(net, testPath, device, os.path.basename(testPath), pred="opts")






