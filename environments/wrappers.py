"""
wrappers.py
-----------
"""

from typing import (
    TypedDict,
    Optional,
    Dict,
    List
)
import copy

import torch
from environments.base import (
    SubsetEnv,
    HistoryEnv
)


class Trajectory(TypedDict):
    initial_state: torch.Tensor
    actions: torch.IntTensor
    terminal_state: torch.Tensor

class Wrapper:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

class TrajectoryRecorder(Wrapper):
    def __init__(self, env: SubsetEnv, termination_key: str = "terminated"):
        """Records trajectories leading to terminal states.

        Args:
            termination_key (str, optional): Key which is used to identify terminal states from the observation dict. Defaults to "terminated".
        """
        super().__init__(env)
        self._initial_state = self._env.loc.clone()
        self._trajectories: List[Trajectory] = []
        self._solutions = []
        self._termination_key = termination_key

    @property
    def terminal_trajectories(self) -> List[Trajectory]: return self._trajectories

    @property
    def solutions(self) -> List[torch.Tensor]: return self._solutions

    def reset(self, *args, **kwargs):
        ret = self._env.reset(*args, **kwargs)
        self._initial_state = self._env.loc.clone()
        return ret

    def masked_reset(self, reset_mask: torch.BoolTensor):
        ret = self._env.masked_reset(reset_mask)
        self._initial_state = self._initial_state*(~reset_mask[..., torch.newaxis]) + \
            self._env.loc*reset_mask[..., torch.newaxis]
        return ret

    def step(self, action: torch.IntTensor) -> Dict:
        ret = self._env.step(action)

        if torch.any(ret[self._termination_key]):
            obs = ret["obs"]
            self._solutions += [obs[c].detach().cpu().numpy() for c in torch.where(ret[self._termination_key])[0]]
            new_trajectories = [Trajectory(
                initial_state   = self._initial_state[c].clone(),
                actions         = self._list_of_actions[:self._running_episode_len.type(torch.int64)[c],c],
                terminal_state  = obs[c].detach().cpu().numpy()) for c in torch.where(ret[self._termination_key])[0]]
            self._trajectories += new_trajectories

        return ret

class EpisodeStatistics(Wrapper):
    def __init__(self, env: HistoryEnv):
        super().__init__(env)

        self._running_total_return = torch.zeros((self._env.num_envs,)).to(self._env.device)
        self._total_return = torch.zeros((self._env.num_envs,)).to(self._env.device)

    def _get_info(self) -> Dict:
        info = self._env._get_info()
        return info | {
            "running_total_return": self._running_total_return,
            "total_return":         self._total_return
        }

    def reset(self, *args, **kwargs):
        ret = self._env.reset(*args, **kwargs)
        self._running_total_return = torch.zeros((self._num_envs,)).to(self._device)
        ret["info"] |= {
            "running_total_return": self._running_total_return,
            "total_return":         self._total_return
        }
        return ret

    def masked_reset(self, reset_mask: torch.BoolTensor):
        ret = self._env.masked_reset(reset_mask)

        self._total_return = self._running_total_return*reset_mask + self._total_return*(~reset_mask)
        self._running_total_return = self._running_total_return*(~reset_mask)
        return ret

    def step(self, action: torch.IntTensor) -> Dict:
        ret = self._env.step(action)
        self._running_total_return += ret["reward"]
        ret["info"] |= {
            "running_total_return": self._running_total_return,
            "total_return":         self._total_return
        }
        return ret

class Autoreset(Wrapper):
    def __init__(self, env: HistoryEnv, auxiliary_keys: Optional[List[str]] = None):
        super().__init__(env)
        self._auxiliary_keys = auxiliary_keys if auxiliary_keys else env._aux_keys

    def step(self, action: torch.IntTensor) -> Dict:
        ret = self._env.step(action)
        real_next_obs = self._env._get_obs()
        aux_data = {f"real_next_{aux}": copy.deepcopy(ret[aux]) for aux in self._auxiliary_keys}
        reset_mask = torch.logical_or(ret["terminated"], ret["truncated"])
        if torch.any(reset_mask):
            if self._auxiliary_keys is []:
                self._env.masked_reset(reset_mask)
            else:
                # Update auxiliary keys which have been reset
                new_aux_data = self._env.masked_reset(reset_mask)
                for k in self._auxiliary_keys:
                    for i, j in enumerate(torch.where(reset_mask)[0]):
                        ret[k][j] = new_aux_data[k][i]
        ret["obs"] = self._get_obs()
        info = self._get_info()
        info["real_next_obs"] = real_next_obs
        info |= aux_data

        ret["info"] = info
        return ret


class DictToList(Wrapper):
    def __init__(self, env: HistoryEnv, auxiliary_keys: Optional[List[str]] = None):
        """Converts all returned objects to lists.

        Args:
            auxiliary_keys (Optional[List[str]], optional): List of additional keys to extract from the returned dictionary.
                the auxiliary data will be appended to the returned tuples.
        """
        super().__init__(env)
        self._auxiliary_keys = auxiliary_keys if auxiliary_keys else env._aux_keys

    def reset(self, *args, **kwargs) -> List:
        ret = self._env.reset(*args, **kwargs)
        return ret["obs"], ret["info"], *[ret[aux] for aux in self._auxiliary_keys]

    def step(self, action: torch.IntTensor) -> List:
        ret = self._env.step(action)
        return ret["obs"], ret["reward"], ret["terminated"], ret["truncated"], ret["info"], *[ret[aux] for aux in self._auxiliary_keys]
