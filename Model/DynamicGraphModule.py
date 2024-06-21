import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from Model.weighted_GCN import weighted_GCN


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


class DynamicGraph(nn.Module):
    def __init__(self, total_num, embedding_label, hidden_dimension, output_dim, layers, device, model_way,
                 embedding_text, dropout=0):
        super(DynamicGraph, self).__init__()
        self.embedding_label = embedding_label
        self.embedding_text = embedding_text
        self.hidden_dimension = hidden_dimension
        self.layers = layers
        self.output_dim = output_dim

        self.model_way = model_way

        self.total_num = total_num
        self.device = device
        self.type_text = ["text", "label"]

        self.Embeddings = torch.nn.Embedding(num_embeddings=total_num, embedding_dim=embedding_label)
        nn.init.xavier_uniform_(self.Embeddings.weight)

        self.attention = weighted_GCN(
            in_features={t: in_dim for t, in_dim in zip(self.type_text, [embedding_text, self.embedding_label])},
            hidden_sizes={t: [hidden_dimension for l in range(layers - 1)] for t in
                          self.type_text},
            out_features={t: hidden_dimension for t in self.type_text}, type_feat=self.type_text,
            device=self.device)

        input_dim = hidden_dimension
        self.adap = nn.Linear(2, 1, bias=False)
        self.fusion = MLLinear(linear_size=[input_dim, input_dim], output_size=output_dim)

        self.dropout = nn.Dropout(dropout)
        self.one_hots = torch.eye(total_num)

    def init_weights(self):
        nn.init.xavier_uniform_(self.Embeddings.weight)

    def k_hots(self, indexs):
        tensors = []
        for index in indexs:
            res = torch.sum(self.one_hots[index], dim=0)
            tensors.append(res)
        tensors = torch.stack(tensors, dim=0).to(self.device)
        return tensors

    def Embedding(self, ids, clacs_len, embedding, type='text'):
        """
        Args:
            ids: index of embedding: [tensor1,...,tensorN]
            clacs_len: [len1,...lenN]
            embedding: M*F
        """

        pad_set_embed_seqs = pad_sequence(ids, batch_first=True, padding_value=0)
        pad_set_embed_seqs = pad_set_embed_seqs.to(self.device)
        input_emb = embedding[pad_set_embed_seqs]  # N * Max_L * F
        embed_input_x_packed = pack_padded_sequence(input_emb, clacs_len, batch_first=True, enforce_sorted=False)
        encoder_outputs, lens_unpacked = pad_packed_sequence(embed_input_x_packed, batch_first=True)

        if type == 'text':
            lens_unpacked = lens_unpacked.unsqueeze(-1).to(self.device)
            out = torch.sum(encoder_outputs, dim=1) / lens_unpacked  # label_feature: N * F
        else:
            out = torch.sum(encoder_outputs, dim=1)
        return self.dropout(out)

    def fetchFeature(self, textIds, labelIds, textLen, labelLen, embedding_Word, embedding_label=None):
        text_out = self.Embedding(textIds, textLen, embedding_Word.weight, type='text')
        label_out = self.Embedding(labelIds, labelLen,
                                   0.1 * embedding_label + self.Embeddings.weight if embedding_label is not None else self.Embeddings.weight,
                                   type='label')
        return {
            "text": text_out,
            "label": label_out
        }

    def mean_operation(self, data, lengths):
        pad_set_embed_seqs = pad_sequence(data, batch_first=True, padding_value=0)
        pad_set_label_seqs = pad_set_embed_seqs.to(self.device)

        text_input_x_packed = pack_padded_sequence(pad_set_label_seqs, lengths, batch_first=True, enforce_sorted=False)
        encoder_outputs, lens_unpacked = pad_packed_sequence(text_input_x_packed, batch_first=True)

        lens_unpacked = lens_unpacked.unsqueeze(-1).to(self.device)
        out = torch.sum(encoder_outputs, dim=1) / lens_unpacked  # label_feature: N * F
        return out

    def lstm_operation(self, data, lengths, type_d, lstm_model):
        pad_set_embed_seqs = pad_sequence(data, batch_first=True, padding_value=0)
        pad_set_label_seqs = pad_set_embed_seqs.to(self.device)

        text_input_x_packed = pack_padded_sequence(pad_set_label_seqs, lengths, batch_first=True, enforce_sorted=False)
        outs, (hidden, _) = lstm_model[type_d](text_input_x_packed)
        out = hidden[0]

        return out

    def forward(self, graphs: dgl.DGLGraph, embedding_Word, weights, node_feats, node_lens, lengths,
                sameMatirx=None) -> torch.Tensor:
        B = len(lengths)
        init_feats = self.fetchFeature(textIds=node_feats['text'], labelIds=node_feats['label'],
                                       textLen=node_lens['text'], labelLen=node_lens['label'],
                                       embedding_Word=embedding_Word, embedding_label=sameMatirx)
        assert init_feats['text'].shape == torch.Size([sum(lengths), self.embedding_text])
        # assert init_feats['label'].shape == torch.Size([sum(lengths), self.total_num])
        assert init_feats['label'].shape == torch.Size([sum(lengths), self.embedding_label])

        readout_feats = dict()

        ids, start = [], 0
        identities = []
        for length in lengths:
            end = start + length
            identities.extend(range(length))
            start = end
        identities = torch.tensor(identities).to(self.device)
        if self.model_way == 'gcn':
            all_feats, readout_feats = self.attention(graphs, init_feats, weights, identities)

        res_emb = self.fusion(
                    self.adap(torch.stack([readout_feats['text'], readout_feats['label']], dim=-1)).squeeze(dim=-1))
        assert res_emb.shape == torch.Size([B, self.output_dim])
        return res_emb
