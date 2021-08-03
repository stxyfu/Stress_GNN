# Refer to the example models from pytorch_geometric
# https://github.com/rusty1s/pytorch_geometric/tree/master/examples

import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, SAGEConv, AGNNConv, GENConv, GATConv
from torch_geometric.nn import BatchNorm, PairNorm, DeepGCNLayer, GraphUNet
from torch import nn, cat, tanh, relu
from torch_geometric.utils import dropout_adj
from torch.nn import Linear, LayerNorm, ReLU, MaxPool1d, BatchNorm1d


# multiple-layer GCN
class StressGCN_Conv(nn.Module):
    def __init__(self, in_channels, hidden_channels, layerNum, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()

        self.layerNum = layerNum
        self.node_encoder = Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        self.batchNorms = nn.ModuleList()

        for i in range(self.layerNum):
            conv = GCNConv(hidden_channels, hidden_channels, improved=improved, cached=cached, bias=bias)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            self.batchNorms.append(norm)
            self.layers.append(conv)

        self.lin1 = Linear(hidden_channels, int(hidden_channels/2.0))
        self.lin2 = Linear(int(hidden_channels / 2.0), 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, batch_norm in zip(self.layers, self.batchNorms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        return self.lin2(ReLU(inplace=True)(self.lin1(x)))



# DeepGCN
class StressDeepGCN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers, dropout=0.1):
        super().__init__()
        self.node_encoder = Linear(input_channels, hidden_channels)
        self.p = dropout

        self.layers = nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=self.p,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, output_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.p, training=self.training)

        return self.lin(x)

# 8-layer GAT
# net = models.StressGCN_GAT_8layer(18, 64)
class StressGCN_GAT_8layer(nn.Module):
    def __init__(self, in_channels, hidden_channels, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()

        self.layerNum = 8
        headNum = 4
        self.node_encoder = Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        self.batchNorms = nn.ModuleList()

        conv = GATConv(hidden_channels, hidden_channels, heads=headNum)
        norm = LayerNorm(hidden_channels * headNum, elementwise_affine=True)
        self.batchNorms.append(norm)
        self.layers.append(conv)

        for i in range(1, self.layerNum):
            conv = GATConv(hidden_channels * headNum, hidden_channels, heads=headNum)
            norm = LayerNorm(hidden_channels * headNum, elementwise_affine=True)
            self.batchNorms.append(norm)
            self.layers.append(conv)

        self.lin1 = Linear(hidden_channels * headNum, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, batch_norm in zip(self.layers, self.batchNorms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        return self.lin2(ReLU(inplace=True)(self.lin1(x)))

# gUNet(256,64,1，4)
class StressGCN_UNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, unet_hidden_channels):
        super().__init__()
        layerNum = 4
        pool_ratios = [0.5]
        self.node_encoder = Linear(in_channels, hidden_channels)

        self.unet = GraphUNet(hidden_channels, unet_hidden_channels, 1,
                              depth=layerNum, pool_ratios=pool_ratios)

        self.node_decoder = Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        edge_index, _ = dropout_adj(data.edge_index, p=0.1,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.unet(x, edge_index)
        return x






# # 8-layer GAT
# class StressGCN_GAT_8layer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, improved=False,
#                  cached=False, bias=True, fine_marker_dict=None):
#         super().__init__()
#
#         self.layerNum = 8
#         headNum = 2
#         self.node_encoder = Linear(in_channels, hidden_channels)
#
#         self.layers = nn.ModuleList()
#         self.batchNorms = nn.ModuleList()
#
#         conv = GATConv(hidden_channels, hidden_channels, heads=headNum)
#         norm = LayerNorm(hidden_channels * headNum, elementwise_affine=True)
#         self.batchNorms.append(norm)
#         self.layers.append(conv)
#
#         for i in range(1, self.layerNum):
#             conv = GATConv(hidden_channels * headNum, hidden_channels, heads=headNum)
#             norm = LayerNorm(hidden_channels * headNum, elementwise_affine=True)
#             self.batchNorms.append(norm)
#             self.layers.append(conv)
#
#         self.lin1 = Linear(hidden_channels * headNum, hidden_channels)
#         self.lin2 = Linear(hidden_channels, 1)
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.node_encoder(x)
#         for conv, batch_norm in zip(self.layers, self.batchNorms):
#             x = conv(x, edge_index)
#             x = batch_norm(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.1, training=self.training)
#         return self.lin2(ReLU(inplace=True)(self.lin1(x)))





























