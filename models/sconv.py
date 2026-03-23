from typing import Optional
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import (
    MessagePassing,
    LayerNorm
)
from torch_geometric.utils import softmax


class SConv(MessagePassing):
    def __init__(self,
        embd_dim: int,
        num_heads: int,
        block_size: int,
        **kwargs
    ) -> None:
        kwargs.setdefault("aggr", "mean")
        kwargs.setdefault("node_dim", 0)
        super().__init__(**kwargs)

        self._block_size = block_size
        self._num_heads = num_heads
        self._embd_dim = embd_dim

        self._cross_attn = nn.Linear(embd_dim, 2*embd_dim*num_heads)
        self._query = nn.Linear(embd_dim, embd_dim*num_heads)

        self._mlp = nn.Sequential(
            nn.Linear(embd_dim*block_size, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.IntTensor) -> torch.Tensor:
        return self.propagate(edge_index, x = x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        k, v = self._cross_attn(x_j).chunk(2, dim=-1)
        q = self._query(x_i)

        q = q.view(-1, x_i.shape[1], self._num_heads, self._embd_dim).transpose(1,2)
        k = k.view(-1, x_i.shape[1], self._num_heads, self._embd_dim).transpose(1,2)
        v = v.view(-1, x_i.shape[1], self._num_heads, self._embd_dim).transpose(1,2)

        # attn = F.softmax(((q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))), dim=-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self._embd_dim) # no softmax
        return self._mlp((attn @ v).mean(dim=1).flatten(-2,-1))
