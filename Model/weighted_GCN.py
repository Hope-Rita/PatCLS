import math

import torch.nn as nn
import torch
import dgl
import dgl.function as fn


class weighted_graph_conv(nn.Module):
    """
        Apply graph convolution over an input signal.
    """

    def __init__(self, in_features: int, out_features: int):
        super(weighted_graph_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, graph, node_features, edge_weights):
        r"""Compute weighted graph convolution. ----- Input: graph : DGLGraph, batched graph. node_features :
        torch.Tensor, input features for nodes (n_1+n_2+..., in_features) or (n_1+n_2+..., T, in_features)
        edge_weights : torch.Tensor, input weights for edges  (T, n_1^2+n_2^2+..., n^2)

        Output:
        shape: (N, T, out_features)
        """
        graph = graph.local_var()
        # multi W first to project the features, with bias
        # (N, F) / (N, T, F)
        graph.ndata['n'] = node_features
        # edge_weights, shape (T, N^2)
        # one way: use dgl.function is faster and less requirement of GPU memory
        graph.edata['e'] = edge_weights.t().unsqueeze(dim=-1)  # (E, T, 1)
        graph.update_all(fn.u_mul_e('n', 'e', 'msg'), fn.sum('msg', 'h'))

        # another way: use user defined function, needs more GPU memory
        # graph.edata['e'] = edge_weights.t()
        # graph.update_all(self.gcn_message, self.gcn_reduce)
        readouts = dgl.mean_nodes(graph, 'h')
        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        readouts = self.linear(readouts)

        return output, readouts

    @staticmethod
    def gcn_message(edges):
        if edges.src['n'].dim() == 2:
            # (E, T, 1) (E, 1, F),  matmul ->  matmul (E, T, F)
            return {'msg': torch.matmul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'].unsqueeze(dim=1))}

        elif edges.src['n'].dim() == 3:
            # (E, T, 1) (E, T, F),  mul -> (E, T, F)
            return {'msg': torch.mul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'])}

        else:
            raise ValueError(f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}")

    @staticmethod
    def gcn_reduce(nodes):
        # propagate, the first dimension is nodes num in a batch
        # h, tensor, shape, (N, neighbors, T, F) -> (N, T, F)
        return {'h': torch.sum(nodes.mailbox['msg'], 1)}

class PositionalEncoding(nn.Module):
    def __init__(self, time_dim, device, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, time_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, time_dim, 2).float() * (-math.log(10000.0) / time_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.to(device)


    def forward(self, ids):
        return self.pe[ids, :]


class weighted_GCN(nn.Module):
    def __init__(self, in_features: dict, hidden_sizes: dict, out_features: dict, type_feat: list, dropout=0.5, device='cpu'):
        super(weighted_GCN, self).__init__()
        self.type_feat = type_feat
        self.positional_Emb = nn.ModuleDict({t: PositionalEncoding(in_features[t], device) for t in type_feat})
        gcns, relus, bns, linear = nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict()
        transform_q, transform_k, transform_v, transform_F = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ffn = nn.ModuleList()
        # layers for hidden_size
        self.lay_nums = len(hidden_sizes[type_feat[0]])+1
        for i, t in enumerate(type_feat):
            input_size = in_features[t]
            for hidden_size in hidden_sizes[t]:
                gcns[t].append(weighted_graph_conv(input_size, hidden_size))
                relus[t].append(nn.ReLU())
                bns[t].append(nn.BatchNorm1d(hidden_size))
                input_size = hidden_size
            # output layer
            gcns[t].append(weighted_graph_conv(hidden_sizes[t][-1], out_features[t]))
            relus[t].append(nn.ReLU())
            bns[t].append(nn.BatchNorm1d(out_features[t]))

        self.gcns, self.relus, self.bns, self.transq, self.transk, self.transv, self.transf, self.ffn = \
            gcns, relus, bns, transform_q, transform_k, transform_v, transform_F, ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, node_features: dict, edges_weight: torch.Tensor, identities: torch.Tensor):
        """
        :param graph: a graph
        :param node_features: shape (n_1+n_2+..., n_features)
        :param edges_weight: shape (T, n_1^2+n_2^2+...)
        :return:
        """
        # h = node_features
        h = dict()
        for ntype, n_feat in node_features.items():
            time_encoding = self.positional_Emb[ntype](identities)
            h[ntype] = n_feat + time_encoding
        readouts = dict()
        for layer in range(self.lay_nums):
            for t in self.type_feat:
                gcn, relu, bn = self.gcns[t][layer], self.relus[t][layer], self.bns[t][layer]
                h[t], readout_t = gcn(graph, h[t], edges_weight)
                h[t] = relu(bn(self.dropout(h[t])))
                readouts[t] = readout_t

        return h, readouts


class stacked_weighted_GCN_blocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(stacked_weighted_GCN_blocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, nodes_feature, edge_weights = input
        h = nodes_feature
        for module in self:
            h = module(g, h, edge_weights)
        return h


def get_edges_weight(patents, max_window=30):
    edges_weight_dict = torch.zeros(len(patents), len(patents))
    for i in range(len(patents)):
        edges_weight_dict[i, i] += 1.
        for j in range(i + 1, min(len(patents), i + 1 + max_window)):
            edges_weight_dict[i, j] += 1. / (j - i)
            edges_weight_dict[j, i] += 1. / (j - i)
    return edges_weight_dict
