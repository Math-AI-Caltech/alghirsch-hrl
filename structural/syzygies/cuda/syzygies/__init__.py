from typing import List
import torch
from . import _C

__all__ = ["SGraph", "SGraphCached", "count_irreducible_recompute"]

def count_irreducible_recompute(mask: torch.Tensor, n: int, d: int, max_level: int = 2) -> int:
    return torch.ops.syzygies.count_irreducible_recompute.default(mask, n, d, max_level)

class SGraphCached:
    def __init__(self, n: int, d: int, device = "cpu", max_level: int = 2):
        self._sgraph_ptr = torch.ops.syzygies.create_sgraph(torch.tensor(n, device = device), torch.tensor(d, device = device), max_level)

    @property
    def num_generators(self) -> int:
        return torch.ops.syzygies.num_generators(self._sgraph_ptr)

    def count_irreducible(self, mask: torch.Tensor, **kwargs) -> int:
        return torch.ops.syzygies.count_irreducible(mask, self._sgraph_ptr[0].numpy().tolist())

    def find_irreducible(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ops.syzygies.find_irreducible(mask, self._sgraph_ptr[0].numpy().tolist())

    def __del__(self):
        torch.ops.syzygies.free_sgraph(self._sgraph_ptr)

class SGraph:
    def __init__(self, n: int, d: int, max_level: int = 2):
        self._sgraph = torch.classes.syzygies.SGraph(d, n, max_level)

    def count_irreducible(self, mask: torch.Tensor) -> int:
        return self._sgraph.count_irreducible(mask)
