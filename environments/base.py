from typing import (
    TypedDict,
    Tuple,
    Dict
)

import torch
import torch.nn.functional as F

class Trajectory(TypedDict):
    initial_state: torch.Tensor
    actions: torch.IntTensor
    terminal_state: torch.Tensor

class HistoryEnv:
    r"""Base environment that tracks action history and episode lengths."""
    def __init__(self, num_envs: int, max_len: int, device: str) -> None:
        self._num_envs = num_envs
        self._max_len = max_len
        self._device = device

        self._running_episode_len = torch.zeros((num_envs,), device = device)
        self._episode_len = torch.zeros((num_envs,), device = device)

        self._list_of_actions = torch.zeros(
            (max_len, num_envs), dtype = torch.int64, device = device)
        self._aux_keys = []

    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError("_get_obs is missing.")

    def _get_info(self) -> Dict[str, torch.Tensor]:
        return {
            "running_episode_len":  self._running_episode_len,
            "total_episode_len":    self._episode_len
        }

    @property
    def device(self) -> str: return self._device

    @property
    def device(self) -> str: return self._device

    @property
    def num_envs(self) -> int: return self._num_envs

    def reset(self, **kwargs) -> Dict[str, object]:
        self._running_total_return = torch.zeros(
            (self._num_envs,), device = self._device)
        self._running_episode_len = torch.zeros(
            (self._num_envs,), device = self._device)

        self._list_of_actions = torch.zeros(
            (self._max_len, self._num_envs), dtype = torch.int64, device = self._device)

        return {
            "obs": self._get_obs(),
            "info": self._get_info()
        }

    def masked_reset(self, reset_mask: torch.BoolTensor) -> None:
        self._episode_len = (
            self._running_episode_len * reset_mask + \
            self._episode_len * (~reset_mask))
        self._running_episode_len = (
            self._running_episode_len * (~reset_mask))

        self._list_of_actions = self._list_of_actions * (
            ~reset_mask[torch.newaxis, ...])

    def step(self, action: torch.IntTensor) -> Dict:
        self._list_of_actions.scatter_(
            dim     = 0,
            index   = self._running_episode_len.type(torch.int64).unsqueeze(0),
            src     = action.type(self._list_of_actions.dtype).unsqueeze(0))
        self._running_episode_len += 1

        return {
            "obs": None,
            "reward": None,
            "terminated": None,
            "truncated": self._running_episode_len >= self._max_len,
            "info": self._get_info()
        }

class SubsetEnv(HistoryEnv):
    def __init__(self, num_envs: int, max_len: int, dim: int, device: str) -> None:
        super().__init__(
            num_envs = num_envs,
            max_len  = max_len,
            device   = device)

        self._dim = dim
        self._loc = torch.zeros((num_envs, dim), device = device)

    def _sample(self, num_envs: int) -> torch.Tensor:
        raise NotImplementedError("Attempted to _sample(...) from a base env `SubsetEnv`")

    def _get_obs(self) -> torch.Tensor:
        return self._loc.clone() #TODO: revisit

    @property
    def num_actions(self) -> int: return self._dim

    @property
    def loc(self) -> torch.Tensor: return self._loc

    def reset(self, **kwargs) -> Dict[str, object]:
        loc = kwargs.get("loc")
        _ = super().reset(**kwargs)
        self._loc = self._sample(self._num_envs) if loc is None else loc

        return {
            "obs": self._get_obs(),
            "info": self._get_info()
        }

    def masked_reset(self, reset_mask: torch.BoolTensor) -> None:
        super().masked_reset(reset_mask)
        self._loc = self._loc*(~reset_mask[..., torch.newaxis]) + \
            reset_mask[..., torch.newaxis]*self._sample(self._num_envs)

    def simulate(self, obs: torch.Tensor, action: torch.IntTensor) -> torch.IntTensor:
        return torch.fmod(obs + F.one_hot(action, num_classes = self._dim), 2)

    def step(self, action: torch.IntTensor) -> Dict:
        res = super().step(action)
        self._loc = self.simulate(self._loc, action)

        return res
