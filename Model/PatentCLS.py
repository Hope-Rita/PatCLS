#!/usr/bin/env python3
# -*- coding: utf-8
import dgl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from Model.DynamicGraphModule import DynamicGraph
from Model.Hierarchy_Emb import Hierarchy_Embedding


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, dropout, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.bidrec = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers_num,
                            batch_first=True, bidirectional=bidirectional)
        if self.bidrec:
            self.init_state = nn.Parameter(torch.zeros(2 * 2 * layers_num, 1, hidden_size), requires_grad=False)
        else:
            self.init_state = nn.Parameter(torch.zeros(2 * layers_num, 1, hidden_size), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, **kwargs):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, len(lengths), 1])
        cell_init, hidden_init = init_state[:init_state.size(0) // 2], init_state[init_state.size(0) // 2:]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.tolist(), batch_first=True,
                                                          enforce_sorted=False)
        if self.bidrec:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True)
        else:
            outs, (hidden, _) = self.lstm(packed_inputs, (hidden_init, cell_init))
            outputs = hidden[0]

        return self.dropout(outputs)


class MLAttention(nn.Module):
    def __init__(self, labels_num, hidden_size, device):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor, sameMatrix=None):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        if sameMatrix is not None:
            code_Emb = torch.cat([sameMatrix, self.attention.weight], dim=-1)
            attention = torch.matmul(inputs, code_Emb.t()).transpose(1, 2).masked_fill(~masks, -np.inf)
        else:
            attention = self.attention(inputs).transpose(1, 2).masked_fill(~masks, -np.inf)  # N, labels_num, L
        attention = F.softmax(attention, -1)
        return attention @ inputs  # N, labels_num, hidden_size


class Embedding(nn.Module):
    """

    """

    def __init__(self, device=None, emb_init=None, padding_idx=0,
                 dropout=0.2):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding.from_pretrained(emb_init, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx
        self.device = device

    def forward(self, inputs):
        input_emb = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != 0).sum(dim=-1), inputs != 0

        return input_emb, lengths, masks



class Network(nn.Module):
    def __init__(self, emb_init=None, device=None, padding_idx=0,
                 emb_dropout=0.2,
                 **kwargs):
        super(Network, self).__init__()
        self.device = device
        self.emb = Embedding(device, emb_init, padding_idx, emb_dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionDNY(Network):
    """

    """

    def __init__(self, labels_num, emb_size, hidden_size, emb_init, class_nums, hierarchy_tree, model_way,
                 layers_num, linear_size, dynamic_hidden, dynamic_layer,
                 dropout, use_lstm_text, fuse_layer, device):
        super(AttentionDNY, self).__init__(emb_init=emb_init, device=device, emb_dropout=dropout)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.out_dim = labels_num
        self.embedding_label = hidden_size * 2
        self.hierarchy_dim = self.embedding_label * 2 if self.use_hierarchy else self.embedding_label
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout, bidirectional=True)
        self.attention = MLAttention(labels_num, self.embedding_label, device)
        self.linear = MLLinear([self.hierarchy_dim] + linear_size, 1)
        self.local_decoders = Hierarchy_Embedding(input_dim=self.embedding_label, hidden_dim=self.embedding_label,
                                                      device=device, numbers=class_nums, fuse_layer=fuse_layer,
                                                      hierarchy_tree=hierarchy_tree)

        self.frequencies = DynamicGraph(total_num=labels_num, hidden_dimension=dynamic_hidden,
                                            output_dim=labels_num, layers=dynamic_layer, device=device,
                                            embedding_text=self.embedding_label if use_lstm_text else emb_size,
                                            dropout=dropout, embedding_label=self.hierarchy_dim, model_way=model_way)

    def forward(self, inputs, graphs: [dgl.DGLGraph], weight, node_feats, node_lens,
                lengths, not_initial):
        # process inputs
        B, N = inputs.shape
        emb_out, length_text, mask = self.emb(inputs)
        assert emb_out.shape == torch.Size([B, N, self.emb_size])
        assert mask.shape == torch.Size([B, N])
        rnn_out = self.lstm(emb_out, length_text)  # N, L, hidden_size * 2
        assert rnn_out.shape == torch.Size([B, N, self.hidden_size * 2])

        attn_out = self.attention(rnn_out, mask, None)  # N, labels_num, hidden_size * 2
        assert attn_out.shape == torch.Size([B, self.out_dim, self.embedding_label])
        _, hierarchy_pred = self.local_decoders(rnn_out, mask)
        assert hierarchy_pred.shape == torch.Size([B, self.out_dim, self.embedding_label])
        hirar_predEmb = torch.cat([attn_out, hierarchy_pred], dim=-1)
        pred_attn = self.linear(hirar_predEmb)

        assert pred_attn.shape == torch.Size([B, self.out_dim])
        dynamic_pred = pred_attn.new_zeros(pred_attn.shape)
        dynamic_pred_emb = self.frequencies(graphs, self.emb.emb, weight, node_feats, node_lens, lengths, None)
        dynamic_pred[not_initial.nonzero().flatten()] = dynamic_pred_emb[not_initial.nonzero().flatten()]

        return [torch.sigmoid(dynamic_pred * 0.1 + pred_attn)]

class MLLinear(nn.Module):
    """

    """

    def __init__(self, linear_size, output_size):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(linear_size[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)

