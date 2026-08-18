r"""options.py
"""
from typing import Dict
import torch

from environments.alghirsch import (
    AlgHirschEnv,
    Option
)
from environments.wrappers import Wrapper
from structural.syzygies.search import SearchEngine

class Beta(Wrapper):
    def __init__(self, env: AlgHirschEnv, *,
            max_steps: int = 10,
            num_nodes_threshold: int = 8,
            diam_max: int = 15) -> None:
        super().__init__(env)
        self._max_steps = max_steps
        self._num_nodes_threshold = num_nodes_threshold
        self._diam_bounds = (env._d, diam_max)

    def _threshold_met(self, obs: torch.Tensor, diameter_term: torch.Tensor) -> torch.BoolTensor:
        return torch.logical_and(
            obs.sum(dim = -1) > self._num_nodes_threshold,
            torch.logical_and(diameter_term > self._diam_bounds[0], diameter_term < torch.inf)) # self._diam_bounds[1]??

class HeuristicLinear(Beta):
    def __init__(self, env: AlgHirschEnv, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self._engine = SearchEngine(env._sgraph, device = env.device)

    def _solve(self, obs: torch.Tensor) -> torch.BoolTensor:
        return torch.tensor([
            len(self._engine.search_from(
                    obs[i].cpu().numpy(),
                    max_steps = self._max_steps,
                    diam_bounds = self._diam_bounds,
                    return_on_first_solution = True)) > 0
                for i in range(obs.shape[0])], device = obs.device)

    def step(self, action: torch.IntTensor) -> Dict:
        ret = self._env.step(action)
        obs, diameter_term = ret["obs"], ret["diameter_term"]

        ret["reward"] = torch.zeros_like(ret["reward"])
        ids = torch.where(self._threshold_met(obs, diameter_term))[0]
        if len(ids) > 0:
            solved = self._solve(obs[ids])
            ret["reward"][ids] = solved + 0.
            ret["terminated"][ids] = solved

        return ret

class ChainOption(Beta):
    def __init__(self, env: AlgHirschEnv, **kwargs) -> None:
        super().__init__(env, **kwargs)

    def step(self, action: torch.IntTensor) -> Dict:
        ret = self._env.step(action)
        obs, diameter_term = ret["obs"], ret["diameter_term"]

        self._env._options = torch.logical_or(
            self._options,
            self._threshold_met(obs, diameter_term)).type_as(
                self._options)

        ret["valid_actions"] = self.valid_actions(obs, self.options)
        ret["options"] = self.options

        return ret
