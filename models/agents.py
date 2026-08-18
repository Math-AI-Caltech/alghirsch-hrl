from typing import (
    Optional,
    Tuple,
    List,
    Dict
)
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

from models.sgmodel import SGModel

NEG_INF: float = float('-inf')

class Actor(nn.Module):
    def __init__(self, sgmodel: SGModel) -> None:
        super().__init__()
        self._sgmodel = sgmodel

    def forward(self, x: Dict) -> Tuple:
        return self.get_action(x)

    def get_action(self, x: Dict, valid_actions: Optional[torch.BoolTensor] = None) -> Tuple:
        logits = self._sgmodel(x)[..., 0]
        logits = logits.masked_fill(~valid_actions, NEG_INF) if valid_actions is not None else logits
        probs = Categorical(logits = logits)
        action = probs.sample()
        return action, F.log_softmax(logits, dim = 1), probs.probs, probs.entropy()

class QNetwork(nn.Module):
    def __init__(self, sgmodel: SGModel) -> None:
        super().__init__()
        self._sgmodel = sgmodel

    def forward(self, x: Dict) -> torch.Tensor:
        return self._sgmodel(x)[..., 0]

class MinQNetwork(nn.Module):
    def __init__(self, q_net_1: QNetwork, q_net_2: QNetwork):
        super().__init__()
        self._q_net_1 = q_net_1
        self._q_net_2 = q_net_2

    def forward(self, x: Dict):
        return torch.min(
            self._q_net_1(x),
            self._q_net_2(x))

@dataclass
class Agent:
    policy_spine: Actor
    q_net_spine: MinQNetwork

    policy_spine_optimizer: optim.Optimizer
    q_spine_optimizer: optim.Optimizer

    policy_linear: Actor
    q_net_linear: MinQNetwork

    policy_linear_optimizer: optim.Optimizer
    q_linear_optimizer: optim.Optimizer

@dataclass
class SpineAgent:
    policy: Actor
    q_net: MinQNetwork

    policy_optimizer: optim.Optimizer
    q_optimizer: optim.Optimizer

def get_action(agent: Agent, x: Dict, valid_actions: torch.Tensor, options: torch.Tensor):
    actions_spine, logprobs_spine, *_, entropy_spine = agent.policy_spine.get_action(x, valid_actions = valid_actions)
    values_spine =  (logprobs_spine.exp() * agent.q_net_spine(x)).sum(dim = -1)

    actions_linear, logprobs_linear, *_, entropy_linear = agent.policy_linear.get_action(x, valid_actions = valid_actions)
    values_linear =  (logprobs_linear.exp() * agent.q_net_linear(x)).sum(dim = -1)

    env_ids = torch.arange(actions_spine.shape[0])

    option_ids = options+0 # TODO proper conversion
    actions = torch.stack([actions_spine, actions_linear], dim = 0)
    logprobs = torch.stack([logprobs_spine, logprobs_linear], dim = 0)
    values = torch.stack([values_spine, values_linear], dim = 0)
    entropies = torch.stack([entropy_spine, entropy_linear], dim = 0)

    return {
        "actions": actions[option_ids, env_ids],
        "logprobs": logprobs[option_ids, env_ids, ...],
        "values": values[option_ids, env_ids],
        "entropy": entropies[option_ids, env_ids],
        "entropy_spine": entropy_spine,
        "entropy_linear": entropy_linear,
        "values_spine": values_spine,
        "values_linear": values_linear
    }
