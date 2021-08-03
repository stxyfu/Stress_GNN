import os
import numpy as np
import torch
import builtins
import random

import gzip
import _pickle as cPickle

from torch_geometric.data import Data, Dataset

defaultLogFilePath = "./output.log"
defaultDatasetJsonFilePath = "./datasetPath.json"

def print(*args):
	tmpStr = ""
	for arg in args:
		tmpStr += str(arg)
	tmpStr = tmpStr.rstrip()
	builtins.print(tmpStr)
	with open(defaultLogFilePath, "a+") as file:
		file.writelines(tmpStr + "\n")

def loadDatasetPathList(path=defaultDatasetJsonFilePath):
	import json
	with open(path, 'r') as file:
		data = json.load(file)
	return data["train"], data["validate"], data["test"]

def saveDatasetPathList(trainList, valList, testList, path=defaultDatasetJsonFilePath):
	import json
	tmpDict = {"train":trainList,
			   "validate":valList,
			   "test":testList}
	with open(path, 'w') as file:
		json.dump(tmpDict, file, indent=2)


def load_data_overlapped(path="./data/", ratio=[0.7, 0.15, 0.15]):
	trainList = []
	valList = []
	testList = []
	classes = os.listdir(path)
	for className in classes:
		classDir = os.path.join(path, className)
		tmpFileNames = []
		for filename in os.listdir(classDir):
			if  ".gzip" in filename:
				filePath = os.path.join(classDir, filename)
				tmpFileNames.append(filePath)
		random.shuffle(tmpFileNames)
		train_length = int(len(tmpFileNames) * ratio[0] / sum(ratio))
		valid_length = int(len(tmpFileNames) * ratio[1] / sum(ratio))
		test_length = len(tmpFileNames) - train_length - valid_length
		trainList.extend(tmpFileNames[: train_length])
		valList.extend(tmpFileNames[train_length : train_length + valid_length])
		testList.extend(tmpFileNames[train_length + valid_length :])
	return trainList, valList, testList

def to_sparse_edge_index(x):
	indices = torch.nonzero(x).tolist()
	# backIndices = [[val[1],val[0]] for val in indices]
	# indices = indices.extend(backIndices)
	edge_index = torch.tensor(indices).long()
	edge_index = edge_index.t()
	return edge_index

class trainDataset(Dataset):
	def __init__(self, dataPathList, le="adj", pred="vms"):
		super(trainDataset, self).__init__(dataPathList, transform=None, pre_transform=None)
		self.datasPathList = dataPathList
		self.le = le
		self.pred = pred
		# self.__indices__ = None
	def __len__(self):
		return len(self.datasPathList)
	def get(self, index):
		filePath = self.datasPathList[index]
		with gzip.open(filePath, 'rb') as file:
			tempDict = cPickle.load(file)
		adj = np.array(tempDict[self.le], dtype=int)
		data = np.array(tempDict["chs"], dtype=float)
		label = np.array(tempDict[self.pred], dtype=float)
		adj = torch.from_numpy(adj).long()
		data = torch.from_numpy(data).float()
		label = torch.from_numpy(label).float()
		data = Data(edge_index=adj, x=data, y=label)
		return data
	def _download(self):
		pass
	def _process(self):
		pass

class testDataset(Dataset):
	def __init__(self, dataPathList, le="adj", pred="vms"):
		super(testDataset, self).__init__(dataPathList, transform=None, pre_transform=None)
		self.datasPathList = dataPathList
		self.le = le
		self.pred = pred
		# self.__indices__ = None
	def __len__(self):
		return len(self.datasPathList)
	def get(self, index):
		filePath = self.datasPathList[index]
		with gzip.open(filePath, 'rb') as file:
			tempDict = cPickle.load(file)
		adj = np.array(tempDict[self.le], dtype=int)
		data = np.array(tempDict["chs"], dtype=float)
		label = np.array(tempDict[self.pred], dtype=float)
		nodes = np.array(tempDict["nodes"], dtype=float)
		adj = torch.from_numpy(adj).long()
		data = torch.from_numpy(data).float()
		label = torch.from_numpy(label).float()
		data = Data(edge_index=adj, x=data, y=label, nds = nodes)
		return data
	def _download(self):
		pass
	def _process(self):
		pass

