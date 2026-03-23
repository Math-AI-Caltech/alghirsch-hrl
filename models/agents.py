from typing import (
    Optional,
    Tuple,
    List,
    Dict
)

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

from models.sgmodel import SGModel
# from structural.syzygies.sgraph import SGraph

NEG_INF = float('-inf')

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
