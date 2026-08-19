from __future__ import annotations
from typing import (
    Optional,
    Tuple
)
import heapq
import itertools

import torch
import numpy as np

import diameter
from structural.syzygies.sgraph import SGraph

class SearchEngine:
    def __init__(self, sgraph: SGraph, device: str):
        self._sgraph = sgraph
        self._device = device
        self._cpu_graph = sgraph.graph.cpu()
        self._cost = 0

    def _priority_fn(self, mask: np.ndarray | torch.Tensor) -> int:
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, device = self._device)
        return self._sgraph.count_irreducible(mask.to(self._device) == 1),

    def _num_irr_fn(self, mask: np.ndarray | torch.Tensor):
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, device = self._device)
        return self._sgraph.count_irreducible(mask.to(self._device) == 1)

    def search_from(self, mask: np.ndarray, max_steps: Optional[int] = None, diam_bounds: Optional[Tuple] = None, return_on_first_solution: bool = False):
        visited = dict()
        solutions = []

        if diam_bounds is None:
            diam_bounds = (self._sgraph.d, 1e5)

        subgraph = self._cpu_graph.subgraph(torch.tensor(np.where(mask)[0].tolist()))#.cpu()
        diam = diameter.max_path_len(diameter.shortest_path(subgraph.x, subgraph.edge_index))
        if not (diam > diam_bounds[0] and diam < diam_bounds[1]): return solutions

        counter = itertools.count() # breaks ties
        num_steps = 0

        h_node, *_ = self._priority_fn(mask)
        queue = [(
            h_node,
            next(counter),
            0,
            h_node,
            mask,
            [])]
        heapq.heapify(queue)
        while len(queue) > 0 and (num_steps < max_steps if max_steps is not None else True):
            _, _, g, _, mask, path = heapq.heappop(queue)

            k = ''.join(map(str, mask.tolist()))
            if k in visited:
                continue
            num_steps += 1

            path = path + [mask.tolist()]
            visited[k] = True

            if self._num_irr_fn(mask) == 0:
                solutions.append((mask.tolist(), path))
                if return_on_first_solution:
                    return solutions
                continue

            for i in range(self._sgraph.num_generators):
                new_mask = np.fmod(mask + np.eye(self._sgraph.num_generators)[i], 2)
                if ''.join(map(str, new_mask.tolist())) in visited: continue

                subgraph = self._cpu_graph.subgraph(torch.tensor(np.where(new_mask)[0].tolist()))#.cpu()
                diam = diameter.max_path_len(diameter.shortest_path(subgraph.x, subgraph.edge_index))

                if diam > diam_bounds[0] and diam < diam_bounds[1]:
                    h_node, *_ = self._priority_fn(new_mask)
                    heapq.heappush(queue, (
                        g + self._cost + h_node,    # full cost g(n) + h(n)
                        next(counter),              # to break ties
                        g + self._cost,             # g(n)
                        h_node,                     # heuristic h(n)
                        new_mask,                   # state
                        path))
        return solutions
