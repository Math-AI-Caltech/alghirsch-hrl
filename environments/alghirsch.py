from typing import (
    Tuple,
    Dict,
    List
)
import torch
from torch_geometric.data import Data
import numpy as np

import sys; sys.path.append("../")
from environments.base import SubsetEnv
from structural.syzygies.sgraph import (
    SGraph,
    Backend
)

import diameter

from enum import Enum
class Option(int, Enum):
    SPINE = 0
    LINEAR = 1

# Profiling using `line_profiler`
import sys
maybe_profile=lambda x:x
if "kernprof" in sys.modules: maybe_profile=lambda x:profile(x)

def _diameter_batched_fn(graphs: List[Data], device: str) -> torch.Tensor:
    clip_diam = (lambda x: -float('inf') if x == float('inf') else x)
    safemax = lambda x: x.max() if x.size != 0 else float('inf')

    return torch.tensor([
        clip_diam(
            safemax(
                np.abs(
                    diameter.max_path_len(
                        diameter.shortest_path(g.x, g.edge_index)))))
        for g in map(lambda _g: _g.to("cpu"), graphs)], device = device)

class AlgHirschEnv(SubsetEnv):
    def __init__(self, *,
        num_envs: int,
        n: int,
        d: int,
        device: str,
        max_len: int = None,
        **kwargs
    ) -> None:
        self._n = n
        self._d = d
        self._sgraph = SGraph(
            d = d, n = n, backend = Backend.CUDA, device = device)
        super().__init__(
            num_envs    = num_envs,
            max_len     = max_len if max_len is not None else d + 6,
            dim         = self._sgraph.num_generators,
            device      = device)

        self._options = torch.tensor([
            kwargs.get("options", Option.SPINE)
                for _ in range(num_envs)], device = self._device)

        self._default_node = torch.tensor([1], device = self._device)
        self._graph = self._sgraph.graph

        # PBRS phi(s) = -#irr(s)
        self._gamma = kwargs.get("gamma", 0.99)
        self._lin_term = torch.zeros((num_envs,), device = self._device)

        self._aux_keys.append("valid_actions")
        self._aux_keys.append("irreducible_edges")
        self._aux_keys.append("options")

    def _safe_where(self, x: torch.Tensor) -> torch.Tensor:
        return x if len(x) != 0 else self._default_node

    def _count_irreducible(self, irreducible_edges: List) -> torch.Tensor:
        return torch.tensor([len(e) for e in irreducible_edges],
            dtype = torch.float32, device = self._device)

    def to_graph(self, obs: torch.Tensor) -> List[Data]:
        """
        Args:
            obs (torch.Tensor): Mask of selected generators. (Shape: (num_envs, #generators))

        Returns:
            List[Data]: List of PyG graphs.
        """
        return [self._graph.subgraph(
            self._safe_where(
                torch.where(obs[i])[0])) for i in range(obs.shape[0])]

    def reset(self, **kwargs) -> Tuple:
        ret = super().reset(**kwargs)
        obs = ret["obs"]

        self._options = torch.tensor([
            kwargs.get("options", Option.SPINE)
                for _ in range(self._num_envs)], device = self._device)
        irreducible_edges = [self._sgraph.find_irreducible(obs[i] == 1) for i in range(obs.shape[0])]
        diameter_term = _diameter_batched_fn(self.to_graph(obs), device = self._device)
        self._lin_term = self._count_irreducible(irreducible_edges)

        return {
            "obs": obs,
            "info": ret["info"],
            "irreducible_edges": irreducible_edges,
            "diameter_term": diameter_term,
            "valid_actions": self.valid_actions(obs, self.options),
            "options": self.options
        }

    @property
    def options(self) -> torch.Tensor:
        return self._options.clone()

    def masked_reset(self, reset_mask: torch.Tensor):
        super().masked_reset(reset_mask)

        self._options = self._options*(~reset_mask) + \
            reset_mask*0 # redundant: also will be handled in Autoreset(...) wrapper.

        obs = self._get_obs()

        diameter_term = _diameter_batched_fn(self.to_graph(obs), device = self._device)
        valid_actions = self.valid_actions(obs, self.options)

        ret = {
            "irreducible_edges": [self._sgraph.find_irreducible(obs[i] == 1) for i in torch.where(reset_mask)[0]],
            "diameter_term": [diameter_term[i] for i in torch.where(reset_mask)[0]],
            "valid_actions": [valid_actions[i] for i in torch.where(reset_mask)[0]],
            "options": [self.options[i] for i in torch.where(reset_mask)[0]]
        }

        self._lin_term[reset_mask] = self._count_irreducible(ret["irreducible_edges"])

        return ret

    def _sample(self, num_envs: int) -> torch.Tensor:
        ids = np.random.randint(0, len(self._sgraph.generators), (num_envs,))
        return torch.tensor(np.eye(self._sgraph.num_generators)[ids], dtype = torch.int).to(self._device)

    def valid_actions(self, obs: torch.Tensor, options: torch.Tensor) -> torch.BoolTensor:
        """
        Args:
            obs (torch.Tensor): Mask of selected generators. (Shape: (num_envs, #generators))

        Returns:
            torch.BoolTensor: Mask of valid actions. (Shape: (num_envs, #generators))
        """
        # ideally obs.shape[0] == num_envs, but this relaxes constraint for testing
        diam_a_fn = lambda a: _diameter_batched_fn(
            self.to_graph(
                self.simulate(obs, torch.tensor([a], device = self._device).expand(obs.shape[0]))),
            device = self._device)

        diam = torch.stack([diam_a_fn(a) for a in range(self.num_actions)])
        positive = diam > 0

        spine_valid = torch.logical_and(
            diam > (obs.sum(dim=-1) - 1), positive).T # Option.SPINE

        linear_valid = torch.logical_and(
            diam > self._d, positive).T # Option.LINEAR

        options = options[..., torch.newaxis]
        return (options*linear_valid + (1-options)*spine_valid) > .5 # .5 is safer

    def set_options(self, options: torch.Tensor):
        self._options = options

    @maybe_profile
    def step(self, action: torch.IntTensor) -> Dict:
        ret = super().step(action)
        obs = self._get_obs()

        irreducible_edges = [self._sgraph.find_irreducible(obs[i] == 1) for i in range(obs.shape[0])]
        lin_term_full = self._count_irreducible(irreducible_edges)
        diameter_term = _diameter_batched_fn(self.to_graph(obs), device = self._device)

        terminated = torch.logical_and(diameter_term == self._d + 1, lin_term_full == 0)

        # PBRS + terminal reward
        reward = {
            "reward": (self._lin_term - self._gamma*lin_term_full) + 100.*terminated,
            "terminated": terminated
        }
        self._lin_term = lin_term_full

        valid_actions = self.valid_actions(obs, self.options)
        ret["truncated"] = torch.logical_or(ret["truncated"], valid_actions.sum(dim=1) == 0)

        return {
            "obs": obs,
            "reward": reward["reward"],
            "terminated": reward["terminated"],
            "truncated": ret["truncated"],
            "info": ret["info"],
            "irreducible_edges": irreducible_edges,
            "diameter_term": diameter_term,
            "valid_actions": valid_actions,
            "options": self.options
        }
