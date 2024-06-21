import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Hierarchy_Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, hierarchy_tree, numbers, device, dropout=0.1, fuse_layer=1,
                 ):
        super(Hierarchy_Embedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = 8
        self.device = device
        self.layer = fuse_layer
        self.dropout = nn.Dropout(p=dropout)
        self.numbers = numbers
        self.hierarchy_tree = [tree.to(device) for tree in hierarchy_tree]
        self.tree_layers = len(hierarchy_tree)
        self.clusters = self.rawMatrix()  # 同类之间的关系

        self.NodeEmb = nn.ModuleList(
            [nn.Embedding(embedding_dim=self.hidden_dim, num_embeddings=number) for number in numbers])
        for emb in self.NodeEmb:
            nn.init.xavier_uniform_(emb.weight)

        self.eye_adjs = [torch.eye(i) for i in self.numbers]


        self.multihead_attn = nn.ModuleList(
            [nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout, batch_first=True) for i
             in self.numbers]
        )
        self.fused_linear = nn.Linear(3, 1, bias=False)
        nn.init.xavier_uniform_(self.fused_linear.weight)

        self.stack_layer = nn.ModuleList(
            [nn.Linear(3, 1, bias=False) for i
             in self.numbers]
        )

        for layer_module in self.stack_layer:
            nn.init.xavier_uniform_(layer_module.weight)

    def rawMatrix(self):
        clusters = []
        for hierarchy in self.hierarchy_tree:
            n1, n2 = hierarchy.shape
            adj_eye = torch.eye(n1).to(self.device)
            # cluster = torch.matmul(torch.matmul(hierarchy.transpose(0, 1), adj_eye), hierarchy)  # n2 * n2
            cluster = torch.matmul(hierarchy.transpose(0, 1), hierarchy)  # n2 * n2

            clusters.append(cluster)
            # print(nor_cluster.shape)

        return clusters

    def calculateF(self, embed_self, embed_up=None, embed_down=None, self_adj=None, hierarchy_low=None,
                   msk_mechanism=False, hierarchy_high=None, layer=0):
        # h = torch.mm(embed_self, self.transform[layer].weight.t())
        # ###################Verision 1###################
        # h = embed_self
        # mat_self = torch.mm(h, h.t()).masked_fill(self_adj != 1, -np.inf)
        # msk_self = torch.softmax(mat_self, dim=1)
        # aggred_mat = torch.mm(msk_self, h)
        # return aggred_mat
        # ###################Verision 1###################
        # high level
        h = embed_self
        # aggred_mat = torch.mm(self_adj, h)
        adp_a = torch.mm(h, h.t())
        if msk_mechanism:
            adp_a = adp_a.masked_fill(self_adj != 1, -np.inf)
        soft_e = torch.softmax(adp_a, dim=1)
        aggred_mat = torch.mm(soft_e, h)  # 来自上层的矩阵约束

        attn_msk, embed_neigh = [], []
        if embed_up is not None:
            attn_msk.append(hierarchy_high.t())  # shape: N2*N1
            embed_neigh.append(embed_up)  # shape: N1*F

        attn_msk.append(torch.eye(h.shape[0]).to(self.device))
        embed_neigh.append(h)

        if embed_down is not None:
            attn_msk.append(hierarchy_low)  # N2*N3
            embed_neigh.append(embed_down)  # shape: N3*F

        if len(attn_msk) != 0:
            attn_msk = torch.cat(attn_msk, dim=1)  # shape: N2*(N1+N2+N3)
            embed_neigh = torch.cat(embed_neigh, dim=0)  # shape: (N1+N2+N3)*F
            attn_mat, attn_weights = self.multihead_attn[layer](h.unsqueeze(dim=0),
                                                                embed_neigh.unsqueeze(dim=0),
                                                                embed_neigh.unsqueeze(dim=0),
                                                                attn_mask=attn_msk != 1)
            attn_mat = attn_mat.squeeze(dim=0)
        else:
            attn_mat = aggred_mat.new_zeros(h.shape)
        # return h
        # return aggred_mat + 0.1 * attn_mat
        return self.stack_layer[layer](torch.stack([aggred_mat, h, attn_mat], dim=-1)).squeeze(dim=-1)


    def filterEdge(self, data: torch.Tensor, filter=0.6):
        zero_vec = data.new_zeros(data.shape)
        top_adpsk = torch.where(data < filter, zero_vec, data)
        return top_adpsk

    def forward(self, inputs: torch.Tensor, masks) -> [[torch.Tensor], torch.Tensor]:
        """
        Args:
            inputs: [batch, N, input_dim]
        Returns:
            out: [batch, output_dim]
        """
        B, L, _ = inputs.shape
        masks = torch.unsqueeze(masks, 1)  # N, 1, L

        #
        top_Rawadp = [self.filterEdge(
            torch.softmax(torch.relu(torch.matmul(self.NodeEmb[i].weight, self.NodeEmb[i].weight.t())), dim=1))
            for i in range(len(self.numbers))]
        # add_adps = [mat if i == 0 else mat * 0.1 + self.clusters[i - 1] for i, mat in enumerate(top_Rawadp)]
        add_adps = [mat if i == 0 else self.clusters[i - 1] for i, mat in enumerate(top_Rawadp)]
        raw_emb = [self.NodeEmb[i].weight for i in range(len(self.numbers))]

        for layer in range(self.layer):
            new_embs = []
            for conv_l in range(len(self.numbers)):
                new_emb = self.calculateF(raw_emb[conv_l],
                                          embed_up=raw_emb[conv_l - 1] if conv_l > 0 else None,
                                          embed_down=raw_emb[conv_l + 1] if conv_l + 1 < len(self.numbers) else None,
                                          self_adj=add_adps[conv_l],
                                          hierarchy_high=self.hierarchy_tree[conv_l - 1] if conv_l > 0 else None,
                                          hierarchy_low=self.hierarchy_tree[
                                              conv_l] if conv_l + 1 < len(self.numbers) else None,
                                          msk_mechanism=conv_l != 0,  # todo:只用第三层，且矩阵为自学习
                                          layer=conv_l)
                new_embs.append(new_emb)
            raw_emb = new_embs
        #
        reflect_one2two = torch.mm(self.hierarchy_tree[0].t(), raw_emb[0])
        reflect_one2thr = torch.mm(self.hierarchy_tree[1].t(), reflect_one2two)

        reflect_two2thr = torch.mm(self.hierarchy_tree[1].t(), raw_emb[1])

        features = self.fused_linear(
            self.dropout(torch.stack([reflect_one2thr, reflect_two2thr, raw_emb[-1]], dim=-1))).squeeze(dim=-1)
        # features = raw_emb[-1]
        attention = (torch.matmul(inputs, features.transpose(0, 1)).transpose(1, 2)).masked_fill(~masks, -np.inf)
        attention = F.softmax(attention, -1)
        attn_out = torch.matmul(attention, inputs)  # N, labels_num, hidden_size

        return features, attn_out

