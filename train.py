import torch
import time, datetime
import models, utils
from utils import print
from torch.optim.lr_scheduler import StepLR
import datetime
defaultProgressViewLoop = 1000
import numpy as np

def train_model(net, data_loader, epochs, lr, device, progressViewLoop=defaultProgressViewLoop):
	startTime = time.time()
	net.train()
	net = net.to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr = lr)
	# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
	# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	for epoch in range(epochs):
		running_loss = 0.0
		progressed_items = 0
		for idx, batch in enumerate(data_loader):
			batch = batch.to(device)
			optimizer.zero_grad()
			outputs = net(batch)
			outputs= outputs.view(-1)
			loss = criterion(outputs, batch.y)
			loss.backward()
			optimizer.step()  # update module
			running_loss += loss.item()
			progressed_items += batch.num_graphs
			if idx % progressViewLoop == progressViewLoop-1:
				print("\n[epoch:%d, batch:%5d] loss: %.8f     (time elapsed: %3f)" %
					  (epoch + 1, idx + 1, running_loss / progressed_items,
					   time.time() - startTime))
		running_loss /= len(data_loader.dataset)
		print("\n[epoch:%d] - finished, loss: %.8f     (time elapsed: %3f  time:%s)" %
			  (epoch + 1, running_loss, time.time() - startTime, datetime.datetime.now()))
		# scheduler.step()
	return net


@torch.no_grad()
def test(net, data_loader, device, status="Test", progressViewLoop=defaultProgressViewLoop):
	net.eval()
	net = net.to(device)
	eval_loss = 0
	progressed_items = 0
	for idx, batch in enumerate(data_loader):
		batch = batch.to(device)
		outputs = net(batch)
		predicted = outputs.view(-1)
		target = batch.y
		eval_loss += mse_loss(predicted, target).item()
		progressed_items += batch.num_graphs
		if idx % progressViewLoop == progressViewLoop - 1:
			print("\n[Test: %d] - loss: %8f" %
				  (idx, float(eval_loss/progressed_items)))
	acc = eval_loss / len(data_loader.dataset)
	print(status + " accuracy: ", acc)

@torch.no_grad()
def test_mape(net, data_loader, device, status="Test", progressViewLoop=defaultProgressViewLoop):
	net.eval()
	net = net.to(device)
	eval_loss = 0
	progressed_items = 0
	for idx, batch in enumerate(data_loader):
		batch = batch.to(device)
		outputs = net(batch)
		predicted = outputs.view(-1)
		target = batch.y
		eval_loss += mape_loss(predicted, target, device).item()
		progressed_items += batch.num_graphs
		if idx % progressViewLoop == progressViewLoop - 1:
			print("\n[Test: %d] - loss: %8f" %
				  (idx, float(eval_loss/progressed_items)))
	acc = eval_loss / len(data_loader.dataset)
	print(status + " accuracy: ", acc)

@torch.no_grad()
def testDraw(net, data_loader, device, status="Test", progressViewLoop=defaultProgressViewLoop):
	from numpy import absolute
	net.eval()
	net = net.to(device)
	for idx, batch in enumerate(data_loader):
		batch = batch.to(device)
		outputs = net(batch)
		predicted = outputs.view(-1)

		input = batch.x.cpu().detach().numpy()
		labels = batch.y.cpu().detach().numpy()
		nodes = batch.nds[0]
		predicted = predicted.cpu().detach().numpy()
		from funcs import plot
		plot.drawMesh(input, nodes, labels)
		plot.drawMesh(input, nodes, predicted)
		plot.drawMesh(input, nodes, absolute(labels-predicted))

		target = batch.y
		eval_loss = mse_loss(predicted, target).item()
		if idx % progressViewLoop == progressViewLoop - 1:
			print("\n[Test: %d] - loss: %8f" %
				  (idx, float(eval_loss)))

@torch.no_grad()
def testDrawGen(net, path, device, label=None, pred="vms"):
	from torch_geometric.data import DataLoader
	from numpy import absolute

	testList = [path]
	testdata = utils.testDataset(testList, pred=pred)
	testloader = DataLoader(testdata, batch_size=1, shuffle=True, num_workers=16)

	net.eval()
	net = net.to(device)
	for idx, batch in enumerate(testloader):
		batch = batch.to(device)
		outputs = net(batch)
		predicted = outputs.view(-1)
		target = batch.y

		input = batch.x.cpu().detach().numpy()
		labels = batch.y.cpu().detach().numpy()
		nodes = batch.nds[0]
		pred = predicted.cpu().detach().numpy()
		from funcs import plot
		maxValue = float(np.max(labels))
		plot.drawMesh(input, nodes, labels, maxValue, "Ground truth", label, True)
		plot.drawMesh(input, nodes, pred, maxValue, "Prediction", label, True)
		plot.drawMesh(input, nodes, absolute(labels-pred), None,"Error", label, True)

		eval_loss = mse_loss(predicted, target).item()
		print("\n[Test: %d] - acc: %8f" %
			  (idx, float(eval_loss)))

def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()

def mse_loss(input, target):
	return ((input - target) ** 2).mean()

def mape_loss(input, target, device, min = 0.01):
	epsilon = min * torch.ones(target.shape[0]).to(device)
	return (torch.abs(input - target) / (target + epsilon)).mean() * 100.0

def rel_mape_loss(input, target):
	stack = torch.stack([input, target], dim=0)
	max = torch.max(stack, dim=0)[0]
	return (torch.abs(input - target) / max).mean() * 100.0
