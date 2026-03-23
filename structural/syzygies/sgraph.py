from typing import (
    Optional,
    List,
    Any
)
from enum import Enum

import numpy as np
from torch_geometric.data import Data

from ._sgraph_py import (
    SGraphPy,
    from_sgraph
)

class Backend(str, Enum):
    PYTHON = "python"
    CUDA = "cuda"

class SGraph:
    """TODO"""
    def __init__(
        self,
        d: int,
        n: int,
        backend: Backend = Backend.PYTHON,
        device: Optional[str] = None,
        max_level: int = 2,
        **kwargs: Any,
    ) -> None:
        self._d = d
        self._n = n
        self._max_level = max_level
        self._device = device
        self._backend = backend
        self._py_sgraph = SGraphPy(d = d, n = n, max_level = max_level, **kwargs)

        if backend == Backend.PYTHON:
            self._impl = self._py_sgraph
        elif backend == Backend.CUDA:
            from syzygies import SGraphCached
            device = "cpu" if device is None else device
            self._impl = SGraphCached(n = n, d = d, device = device, max_level = max_level)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @property
    def graph(self) -> Data:
        return from_sgraph(
            self._py_sgraph.graph, n = self._n, generators = self._py_sgraph.generators, levels = [1]).to(self._device)

    @property
    def generators(self) -> List[int]: return self._py_sgraph.generators

    @property
    def num_generators(self) -> int: return self._impl.num_generators

    @property
    def backend(self) -> Backend: return self._backend

    @property
    def d(self) -> int: return self._d

    @property
    def n(self) -> int: return self._n

    def count_irreducible(self, *args: Any, **kwargs: Any):
        args = self._normalize_indices_args(args)
        if self.backend == Backend.CUDA and kwargs.get("export_edges", False):
            return self._impl.find_irreducible(*args, **kwargs)

        return self._impl.count_irreducible(*args, **kwargs)

    def find_irreducible(self, *args: Any, **kwargs: Any):
        fn = getattr(self._impl, "find_irreducible", None)
        if fn is None:
            raise NotImplementedError(
                "find_irreducible is not available for the selected backend")
        args = self._normalize_indices_args(args)
        return fn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    def __repr__(self) -> str:
        return f"SGraph(d={self._d}, n={self._n}, backend={self._backend.value})"

    def _normalize_indices_args(self, args: Any):
        if not args:
            return args

        first, *rest = args

        if self._backend == Backend.CUDA:
            import torch
            if self._is_bool_mask(first):
                if isinstance(first, torch.Tensor):
                    return first.bool().to(self._device), *rest
                return (torch.tensor(first, device = self._device, dtype = torch.bool), *rest)

            mask = torch.zeros(self.num_generators, dtype = torch.bool, device = self._device)
            mask[first] = True
            return (mask, *rest)

        # backend here is python, but might need to check explicitly if more backends are introduced.
        if self._is_bool_mask(first):
            import torch
            if isinstance(first, torch.Tensor):
                idx = torch.nonzero(first, as_tuple = False).view(-1).tolist()
                return (self._impl.subgraph(idx), *rest)

            idx = [i for i, v in enumerate(first) if v]
            return (self._impl.subgraph(idx), *rest)

        return (self._impl.subgraph(first), *rest)

    def _is_bool_mask(self, x: Any) -> bool:
        import torch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        if isinstance(x, torch.Tensor) and x.dtype == torch.bool and x.ndim == 1 and x.numel() == self.num_generators:
            return True
        if isinstance(x, (list, tuple)) and len(x) == self._n and all(isinstance(v, (bool, int)) for v in x):
            return all(v in (0, 1, True, False) for v in x)
        return False


if __name__ == "__main__":
    import numpy as np
    sgraph = SGraph(d = 7, n = 10, device = "cuda:0", backend = Backend.CUDA)
    mask = np.zeros((sgraph.num_generators,))
    ids = [4, 5, 7, 11, 12, 16, 18, 19, 25, 32, 34]
    mask[ids] = 1
    assert (sgraph.count_irreducible(mask == 1, export_edges = True).tolist() == [])
    assert (sgraph.count_irreducible(ids, export_edges = True).tolist() == [])

    import code; code.interact(local = locals())
