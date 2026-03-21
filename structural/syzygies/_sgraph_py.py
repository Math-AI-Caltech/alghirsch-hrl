"""
syzygies.py
-----------
"""
import numpy as np
from typing import (
    Optional,
    Union,
    Dict,
    List,
) 

import torch
from torch_geometric.data import Data

import itertools
from functools import reduce
from collections import defaultdict

import warnings

def lcm(a: int, b: int) -> int: return a | b
def deg(g: int) -> int: return bin(g).count("1")

class SGraphPy:
    r"""Constructs syzygy graph (decomposed into levels) associated to a space of homogeneous square-free monomial ideals.

    Contains tools for studying first syzygy module of monomial ideals.
    """
    def __init__(self, d: int, n: int, **kwargs):
        self._max_level = kwargs.get("max_level", 2)

        self._n = n
        self._d = d

        if self._n >= 2*self._d and self._max_level < self._d:
            warnings.warn("Max level is too small. Highest level generators might be irreducible, but will be ignored.")

        self._generators = [reduce(
            lcm, map(lambda i: 2**i, gen))
            for gen in itertools.combinations(range(n), d)]
        if kwargs.get("sorted", True):
            self._generators = sorted(
                self._generators)
        self._graph = self._precompute(
            generators  = self._generators,
            max_level   = self._max_level)

    def _precompute(self, generators: List, max_level: int = 2) -> Dict:
        edges = {i+1: defaultdict(lambda: {}) for i in range(max_level)}
        for i in range(len(generators)):
            for j in range(i+1, len(generators)):
                # lcm_gens = lcm(generators[i], generators[j])
                e_i = generators[j] & (~generators[i])
                # e_0 = lcm_gens & (~generators[i])
                level = deg(e_i)
                if level > max_level: continue
                e_j = generators[i] & (~generators[j])
                # e_1 = lcm_gens & (~generators[j])

                edges[level][i][j] = {i: e_i, j: e_j}
                edges[level][j][i] = {i: e_i, j: e_j}

        return {k:dict(v) for k,v in edges.items()}

    def _reduce_level(self, s: int, e: int, weight: Dict, adj: Dict, levels: Optional[List] = None):
        if levels is None: levels = [1]

        visited = []
        queue = [(s, weight[s])]

        while len(queue) > 0:
            s, s_gen = queue.pop()
            for k,v in (adj[levels[0]][s].items() if len(levels) == 1 else itertools.chain(*[adj[level][s].items() for level in levels])):
                if k in visited: continue

                # Check if admissible (i.e. degree doesn't change)
                if s_gen | v[s] == s_gen:
                    # Multiply by the extra factor added from s_gen
                    prod = s_gen & (~v[s])
                    if k == e:
                        return True
                    queue.append([k, v[k] | prod])

            visited.append(s)
        return False

    @property
    def n(self) -> int: return self._n

    @property
    def d(self) -> int: return self._d

    @property
    def graph(self) -> Dict: return self._graph

    @property
    def generators(self) -> List[int]: return self._generators

    @property
    def num_generators(self) -> int: return len(self._generators)

    def subgraph(self, indices: List) -> Dict:
        r"""Computes subgraph induced by the list of vertices.

        Args:
            indices (List): List of indices of vertices in the subgraph.

        Returns:
            Dict: subgraph of the total graph.
        """
        res = {}
        for level in self._graph.keys():
            res[level] = {}
            for i1 in range(len(indices)):
                idx_1 = indices[i1]
                if idx_1 not in self._graph[level]: continue
                if idx_1 not in res[level]: res[level][idx_1] = {}

                for i2 in range(i1 + 1, len(indices)):
                    idx_2 = indices[i2]
                    if idx_2 not in self._graph[level][idx_1]: continue

                    res[level][idx_1][idx_2] = \
                        self._graph[level][idx_1][idx_2]

                    if idx_2 not in res[level]: res[level][idx_2] = {}
                    res[level][idx_2][idx_1] = \
                        self._graph[level][idx_2][idx_1]
        return res

    def is_weakly_linear(self, sub: Dict) -> bool:
        r"""Checks if the ideal is linearly presented.

        Args:
            sub (Dict): subgraph associated to the monomial ideal.
        """
        for level in range(2, self._max_level + 1):
            for i in sub[level]:
                for j in sub[level][i]:
                    if i > j: continue
                    if not self._reduce_level(i, j, sub[level][i][j], sub):
                        return False
        return True

    def diameter(self, sub: Dict, level: int = 1, safety: bool = True) -> int:
        r"""
        Computes diameter using Floyd-Warshall algorithm.

        - Note: Might be less efficient than `scipy.sparse.csgraph.shortest_path`.

        Args:
            sub (Dict): subgraph of the SGraph.
            level (int, optional): Level of the subgraph. Defaults to 1.

        Returns:
            int: diameter.
        """
        to_idx = {v:k for k,v in enumerate(sorted(sub[level].keys()))}
        n = len(to_idx)
        dist = np.zeros((n, n)) + np.inf
        for k,v in sub[level].items():
            for e in v.keys():
                dist[to_idx[k]][to_idx[e]] = 1

        for i in range(n):
            dist[i,i] = 0

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])

        safe_round = lambda x: -1 if x == float('inf') else round(x)
        if not safety:
            safe_round = lambda x: x if x == float('inf') else round(x)
        np.fill_diagonal(dist, np.zeros(dist.shape[0],))
        return safe_round(dist.max())

    def count_irreducible(self, sub: Dict, export_edges: bool = False) -> Union[int, List]:
        r"""Counts the number of irreducible generators at all levels:
        s.t.:  2 ≤ levels ≤ max_level

        Args:
            sub (Dict): subgraph associated to the monomial ideal.
            export_edges (bool): If enabled, returns list of irreducible edges.

        Returns:
            int: number of irreducible generators. (if not export_edges)
            List: list of irreducible edges. (if export_edges)
        """
        irreducible = 0 if not export_edges else []
        for level in range(2, self._max_level+1):
            for i in sub[level]:
                for j in sub[level][i]:
                    if i > j: continue
                    if not self._reduce_level(i, j, sub[level][i][j], sub):
                        irreducible += 1 if not export_edges else [[i,j]]
        return irreducible

    def __repr__(self) -> str:
        return f"SGraph(d={self._d}, n={self._n})"

def from_sgraph(graph: Dict, n: int, generators: List[int], levels: List[int] = None) -> Data:
    r"""Converts graph dict to PyG graph.

    Args:
        graph (Dict): sgraph as dict.
        n (int): Number of variables (fixes/pads number of bits)

    Returns:
        Data: PyG Graph with edge_attrs from the features.
    """
    edge_list = []
    edge_attr = []
    for level in (graph.keys() if levels is None else levels):
        for v0 in graph[level]:
            for v1 in graph[level][v0]:
                # if v0 >= v1: continue
                edge_list.append([v0, v1])
                edge_attr.append([level]) # list(graph[level][v0][v1].values()))
    edge_list = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).contiguous()

    return Data(
        x = torch.tensor(
            [*map(lambda x: [0 for _ in range(n - len(bin(x)[2:]))]  + [*map(int, bin(x)[2:])], generators)]),
        edge_index  = edge_list,
        edge_attr   = edge_attr)


if __name__ == "__main__":
    sgraph = SGraphPy(d = 4, n = 7)
    subgraph = sgraph.subgraph(
        [4, 5, 7, 11, 12, 16, 18, 19, 25, 32, 34])
    assert sgraph.diameter(subgraph) == 5
    assert sgraph.count_irreducible(subgraph) == 0
