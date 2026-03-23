from typing import (
    Dict,
    List
)

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.sconv import SConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GPSConv,
    GATv2Conv,
    GCNConv,
    global_mean_pool
)

from structural.syzygies.sgraph import SGraph

def _expand(data: Data, irreducible_edges: List):
    batch = []
    for edges in irreducible_edges:
        edge_index = torch.tensor(edges).swapaxes(0,1).to(data.x.device) if len(edges) > 0 else \
            torch.zeros((2,0)).long().to(data.x.device)
        edge_attr = 2. + torch.zeros((len(edges), 1)).to(data.x.device)
        edge_index = torch.concat([edge_index, data.edge_index], dim = 1)
        edge_attr = torch.concat([edge_attr, data.edge_attr], dim = 0)
        batch.append(Data(
            x = data.x,
            edge_index = edge_index,
            edge_attr = edge_attr))

    data_batch = Batch.from_data_list(batch).to(data.x.device)
    return data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch

class SGModel(nn.Module):
    def __init__(self,
        sgraph: SGraph,
        output_dim: int = 1,
        node_embd_dim: int = 32,
        num_heads: int = 4,
        num_conv: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self._sgraph = sgraph
        self._output_dim = output_dim
        self._embd_dim = node_embd_dim
        self._mean_pooling = kwargs.get("mean_pooling", False)

        self._node_embd = nn.Embedding(
            2, node_embd_dim)
        self._mask_embd = nn.Embedding(
            2, node_embd_dim)

        self._convs = nn.ModuleList()
        self._convs.append(
            SConv(
                embd_dim    = node_embd_dim,
                block_size  = sgraph.n,
                num_heads   = 2))

        for _ in range(num_conv):
            gconv = GATv2Conv(
                in_channels  = node_embd_dim,
                out_channels = node_embd_dim)
            self._convs.append(
                GPSConv(
                    channels  = node_embd_dim,
                    conv      = gconv,
                    norm      = "layer_norm",
                    heads     = num_heads,
                    attn_type = "multihead"))

        self._proj = nn.ModuleList([
            GCNConv(node_embd_dim, node_embd_dim),
            GCNConv(node_embd_dim, node_embd_dim),
            nn.Linear(node_embd_dim, output_dim)])

    def forward(self, x: Dict) -> torch.Tensor:
        output_shape = x["mask"].shape
        x_mask = self._mask_embd(x["mask"].long().flatten().unsqueeze(-1))
        irreducible_edges = x["irreducible_edges"]
        data = self._sgraph.graph

        x, edge_index, edge_attr, batch = _expand(data, irreducible_edges)
        x = self._node_embd(x.long()) + x_mask

        x = self._convs[0](x, edge_index = edge_index)
        for conv in self._convs[1:]:
            x = conv(x, edge_index = edge_index, batch = batch)

        for i, layer in enumerate(self._proj):
            x = layer(x) if isinstance(layer, nn.Linear) else layer(x, edge_index)
            if i < len(self._proj) - 1:
                x = F.gelu(x)

        if self._mean_pooling:
            x_mp = global_mean_pool(x, batch)
            return x.view(*output_shape, -1), x_mp
        return x.view(*output_shape, -1)
