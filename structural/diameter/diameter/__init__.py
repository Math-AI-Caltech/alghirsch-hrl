import torch
from . import _C

__all__ = ["shortest_path", "shortest_path_parallel"]


def max_path_len(shortest_path_mat: torch.Tensor):
    # return (shortest_path_mat - torch.diag(torch.diagonal(shortest_path_mat))).max()
    return shortest_path_mat.fill_diagonal_(0).max()

def shortest_path(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return torch.ops.diameter.shortest_path.default(x, edge_index)

@torch.library.register_fake("diameter::shortest_path")
def _(x, edge_index):
    return torch.empty(x.shape[0], x.shape[0])

def shortest_path_parallel(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return torch.ops.diameter.shortest_path_parallel.default(x, edge_index)

@torch.library.register_fake("diameter::shortest_path_parallel")
def _(x, edge_index):
    return torch.empty(x.shape[0], x.shape[0])
