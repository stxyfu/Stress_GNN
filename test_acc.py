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

    batch_size = 8
    in_channel = 18

    net = models.StressGCN_Conv_8layer(in_channel, 256, 8)
    net.load_state_dict(torch.load("./results/stress_20000"))

    if os.path.exists("datasetPath.json"):
        trainList, valList, testList = utils.loadDatasetPathList()
    else:
        print("datasetPath.json does not exist")
    print("Total training model: ", len(trainList))
    print("Total validation model: ", len(valList))
    print("Total testing model: ", len(testList))

    # Stress prediction
    traindata = utils.trainDataset(trainList)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
    valdata = utils.trainDataset(valList)
    valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
    testdata = utils.trainDataset(testList)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)
    # Topology optimization prediction
    # traindata = utils.trainDataset(trainList, pred="opts")
    # trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
    # valdata = utils.trainDataset(valList, pred="opts")
    # valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
    # testdata = utils.trainDataset(testList, pred="opts")
    # testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)

    # mse
    train.test(net, trainloader, device)
    train.test(net, valloader, device)
    train.test(net, testloader, device)
    # mape
    train.test_mape(net, trainloader, device)
    train.test_mape(net, valloader, device)
    train.test_mape(net, testloader, device)

