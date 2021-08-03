import torch
from torch_geometric.data import DataLoader
import time, datetime
import random, os
import numpy as np
import models, utils, train, models_test
from utils import print
import time

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

	dataFilesPath = "C:/Data2_Conv_BC2_5000"
	in_channel = 18
	batch_size = 8
	lr = 1e-2
	momentum = 0.9
	ratio = [0.7, 0.15, 0.15]
	savemodel = "stressGCN_net"

	if os.path.exists("datasetPath.json"):
		trainList, valList, testList = utils.loadDatasetPathList()
	else:
		trainList, valList, testList = utils.load_data_overlapped(dataFilesPath, ratio)
		# trainList, valList, testList = utils.load_data_overlapped(dataFilesPath, ratio)
		utils.saveDatasetPathList(trainList, valList, testList)
	print("Total training model: ", len(trainList))
	print("Total validation model: ", len(valList))
	print("Total testing model: ", len(testList))

	traindata = utils.trainDataset(trainList)
	trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
	valdata = utils.trainDataset(valList)
	valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
	testdata = utils.trainDataset(testList)
	testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# for le=1:
	# traindata = utils.trainDataset(trainList, "adj2")
	# trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
	# valdata = utils.trainDataset(valList, "adj2")
	# valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# testdata = utils.trainDataset(testList, "adj2")
	# testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# for le=2:
	# traindata = utils.trainDataset(trainList, "adj3")
	# trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
	# valdata = utils.trainDataset(valList, "adj3")
	# valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# testdata = utils.trainDataset(testList, "adj3")
	# testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)


	# Conventional Methods
	# 3-layer GCN(64,64)
	# net = models.StressGCN_Conv(in_channel, 64, 3)
	# 8-layer GCN(256,256)
	# net = models.StressGCN_Conv(in_channel, 256, 8)
	# 3-layer DeepGCN
	# net = models.StressDeepGCN(in_channel, 64, 1, 3)
	# 8-layer DeepGCN
	# net = models.StressDeepGCN(in_channel, 256, 1, 8)
	# 8-layer GAT(256,256)
	# net = models.StressGCN_GAT_8layer(in_channel, 256)
	# gUNet
	# net = models.StressGCN_UNet(18, 32, 64)


	# BOGE method for stress field prediction
	# net = models.StressDeepGCN(18, 64, 1, 3, dropout=0.0)


	# BOGE method for topology optimization prediction
	# Change to "opts" to get the ground truth value of the topology optimization
	# traindata = utils.trainDataset(trainList, pred="opts")
	# trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=16)
	# valdata = utils.trainDataset(valList, pred="opts")
	# valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# testdata = utils.trainDataset(testList, pred="opts")
	# testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=16)
	# net = models.StressDeepGCN(18, 64, 1, 3)

	torch.autograd.set_detect_anomaly(True)

	print("\nTraining begins...")
	print(datetime.datetime.now())
	for i in range(0,500):
		epochs = 1
		net = train.train_model(net, trainloader, epochs, lr, device)
		torch.save(net.state_dict(), savemodel+"_"+str(i))
		net.load_state_dict(torch.load(savemodel))
		print("\n[Begin validation...]")
		train.test(net, valloader, device, status="Valid")
		print("[Begin testing...]")
		print("[Iteration - ", str(i), "]")
		print(datetime.datetime.now())
		train.test(net, testloader, device)

	print("Training ends: ", datetime.datetime.now())