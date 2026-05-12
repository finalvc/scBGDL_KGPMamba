from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from functools import lru_cache


def get_device():
    return torch.device('cpu')


@lru_cache(maxsize=1)
def _get_num_pathways(kegg_path='../Datasets/KEGG_Mapping.npy', default=320):
    if os.path.exists(kegg_path):
        return np.load(kegg_path, allow_pickle=True).item()['num_pathways']
    return default


# ==========================================================
# 纯 PyTorch 版 Mamba
# ==========================================================
class PurePyTorchMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x_c, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x_c = x_c.transpose(1, 2)
        x_c = self.conv1d(x_c)[:, :, :l]
        x_c = x_c.transpose(1, 2)
        x_c = self.activation(x_c)

        x_dbl = self.x_proj(x_c)
        delta, B, C = x_dbl.split(split_size=[1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        ys = []
        h = torch.zeros((b, self.d_inner, self.d_state), device=x.device, dtype=x.dtype)

        for i in range(l):
            delta_i = delta[:, i]
            B_i = B[:, i]
            C_i = C[:, i]
            x_i = x_c[:, i]

            deltaA_i = torch.exp(delta_i.unsqueeze(-1) * A)
            deltaB_i = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)

            h = deltaA_i * h + deltaB_i * x_i.unsqueeze(-1)
            y_i = (h * C_i.unsqueeze(1)).sum(dim=-1) + D * x_i
            ys.append(y_i)

        y = torch.stack(ys, dim=1)
        y = y * self.activation(res)
        out = self.out_proj(y)
        return out


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = PurePyTorchMamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features=None, heads=2, concat=False,
                 return_attention_weights=True, dropout=0., return_dense=True):
        super().__init__()
        self.out_features = out_features if out_features else in_features
        self.return_dense = return_dense
        self.conv = GATv2Conv(in_features, self.out_features, heads=heads, concat=concat)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.out_features)

    def forward(self, x_nodes, x_edges, x_masks):
        x_nodes_af_conv = x_nodes.new_zeros(
            x_nodes.shape[0], x_nodes.shape[1], self.out_features
        )
        x_edges_af_conv = [i for i in range(x_nodes.shape[0])]

        for graph in range(x_nodes.shape[0]):
            if x_masks is None:
                x_out, edge_out = self.conv(
                    x_nodes[graph][0:],
                    x_edges[graph]['indices'],
                    return_attention_weights=True
                )
                x_nodes_af_conv[graph][0:, :] = x_out
                x_edges_af_conv[graph] = edge_out
            else:
                valid_n = int(x_masks[graph].item())
                x_out, edge_out = self.conv(
                    x_nodes[graph][0:valid_n],
                    x_edges[graph]['indices'],
                    return_attention_weights=True
                )
                x_nodes_af_conv[graph][0:valid_n, :] = x_out
                x_edges_af_conv[graph] = edge_out

        x_nodes_af_conv = self.relu(x_nodes_af_conv[:, :, :self.out_features])
        x_nodes_af_conv = self.ln(x_nodes_af_conv)
        x_nodes_af_conv = self.dropout(x_nodes_af_conv[:, :, :self.out_features])

        if self.return_dense:
            adj_matrix = x_nodes.new_zeros(
                x_nodes.shape[0], x_nodes.shape[1], x_nodes.shape[1]
            ).type(torch.float32)

            for i in range(len(x_edges_af_conv)):
                indice, value = x_edges_af_conv[i]
                value = torch.mean(value, dim=1)
                temp = to_dense_adj(indice, edge_attr=value)
                adj_matrix[i][0:temp.shape[1], 0:temp.shape[1]] = temp

            return x_nodes_af_conv, adj_matrix
        else:
            return x_nodes_af_conv, x_edges


class GDL(torch.nn.Module):
    def __init__(self, input_dim):
        super(GDL, self).__init__()

        self.num_pathways = _get_num_pathways()

        self.conv1 = GraphAttentionLayer(input_dim, 512, heads=4, dropout=0., return_dense=False)
        self.conv2 = GraphAttentionLayer(512, 512, heads=4, dropout=0., return_dense=True)

        self.pool_norm = nn.LayerNorm(512)

        self.mamba_net = nn.Sequential(
            MambaBlock(dim=512),
            MambaBlock(dim=512)
        )

        self.norm_final = nn.LayerNorm(512)
        self.head = nn.Linear(512, 1)

        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)

        self.cls = nn.Parameter(torch.empty(1, 1, 512))
        nn.init.trunc_normal_(self.cls, std=0.01)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_pathways + 1, 512) * 0.001)

    def forward(self, x):
        model_device = next(self.parameters()).device

        x_nodes = x['batch_nodes'].to(model_device)
        x_masks = x['batch_masks'].to(model_device)
        x_edges = x['batch_edges']
        batch_S = x['batch_S'].to(model_device)

        for i in range(len(x_edges)):
            x_edges[i]['indices'] = x_edges[i]['indices'].to(model_device)

        x_nodes, x_edges = self.conv1(x_nodes, x_edges, x_masks)
        x_nodes, adj_matrix = self.conv2(x_nodes, x_edges, x_masks)

        S_sum = batch_S.sum(dim=1, keepdim=True).clamp(min=1.0)
        batch_S_norm = batch_S / S_sum
        pathway_nodes = torch.bmm(batch_S_norm.transpose(1, 2), x_nodes)
        pathway_nodes = self.pool_norm(pathway_nodes)

        cls_token = self.cls.expand(pathway_nodes.shape[0], -1, -1).to(model_device)
        pos_embed = self.pos_embed[:, :pathway_nodes.shape[1] + 1, :].to(model_device)

        x_nodes = torch.cat([pathway_nodes, cls_token], dim=1)
        x_nodes = x_nodes + pos_embed

        x_nodes = self.mamba_net(x_nodes)

        latent_z = self.norm_final(x_nodes[:, -1, :])
        pred = self.head(latent_z).squeeze(dim=1)
        pred = torch.clamp(pred, min=-8.0, max=8.0)

        return pred, latent_z