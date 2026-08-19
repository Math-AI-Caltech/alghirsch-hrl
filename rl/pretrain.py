from __future__ import annotations
from typing import List
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from environments.alghirsch import Option

from models.agents import (
    NEG_INF,
    Actor,
    Agent
)
import numpy as np

@dataclass
class PretrainConfig:
    epochs: int = 250
    num_states: int = 256
    rollout_steps: int = 400
    temp: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    minibatch_size: int = 128
    max_grad_norm: float = 0.5

@dataclass
class States:
    masks: torch.Tensor
    irreducible_edges: List
    valid_actions: torch.Tensor

def _sel(arr,ids): # type: ignore
    return [arr[i] for i in ids]

def collect_states(envs, cfg: PretrainConfig, option: Option = Option.LINEAR) -> States | None:
    r"""Samples states which occur in a specified option.

    Args:
        option (Option, optional): Specified option. Defaults to Option.LINEAR.

    Returns:
        States | None: List of states occuring under specified option. Returns None if such states are never encountered.
    """
    ret = envs.reset()
    masks, irreducible_edges, valid_actions = [], [], []

    with tqdm.tqdm(total = cfg.num_states, desc = "Collecting states") as count_states:
        for _ in range(cfg.rollout_steps):
            if len(masks) >= cfg.num_states: break

            # actions = torch.multinomial(ret["valid_actions"].float(), 1).squeeze()
            probs = ret["valid_actions"].float()
            probs = torch.where(probs.sum(dim = -1, keepdim = True) > 0, probs, torch.ones_like(probs))
            # ret = envs.step(actions)
            ret = envs.step(torch.multinomial(probs, 1).squeeze(dim = -1))

            ids = torch.where(
                torch.logical_and(
                    ret["options"] == option,
                    ret["valid_actions"].sum(dim = -1) > 0))[0]
            for i in ids.tolist():
                masks.append(ret["obs"][i])
                irreducible_edges.append(ret["irreducible_edges"][i])
                valid_actions.append(ret["valid_actions"][i])
            count_states.update(len(ids))

    num_states = min(len(masks), cfg.num_states)
    return States(
        masks               = torch.stack(masks[:num_states]),
        irreducible_edges   = irreducible_edges[:num_states],
        valid_actions       = torch.stack(valid_actions[:num_states])) if num_states > 0 else None

def heuristic_greedy_logprobs(envs, states: States, cfg: PretrainConfig) -> torch.Tensor:
    masks = envs.simulate(
        states.masks.repeat_interleave(envs.num_actions, dim = 0),
        torch.arange(envs.num_actions, device = cfg.device).repeat(states.masks.shape[0]))

    heuristic = torch.tensor(
        [float(envs._sgraph.count_irreducible(mask == 1)) for mask in masks],
        device = cfg.device).view(states.masks.shape[0], envs.num_actions)

    return F.log_softmax(
        (-heuristic / cfg.temp).masked_fill(~states.valid_actions, NEG_INF), dim = -1)

def loss_fn(policy: Actor, states: States, target_logprobs: torch.Tensor, mb_inds: np.ndarray):
    b_valid = states.valid_actions[mb_inds]
    _, b_logprobs, *_ = policy.get_action(
        {"mask": states.masks[mb_inds], "irreducible_edges": _sel(states.irreducible_edges, mb_inds)},
        valid_actions = b_valid)
    b_target_logprobs = target_logprobs[mb_inds]

    return (b_target_logprobs.exp() * (b_target_logprobs - b_logprobs)).masked_fill(~b_valid, 0.).sum(dim = -1).mean() # KL

def pretrain_linear(envs, agent: Agent, cfg: PretrainConfig) -> None:
    r"""Pretrains linear policy.

    Raises:
        RuntimeError: If unable to find any states sampled using linear option.
    """
    states = collect_states(envs, cfg)
    if states is None:
        raise RuntimeError("Unable to construct any states sampled using linear option!")
    num_states = states.masks.shape[0]

    with torch.no_grad():
        target_logprobs = heuristic_greedy_logprobs(envs, states, cfg)

    optimizer = optim.Adam(agent.policy_linear.parameters(), lr = cfg.lr)
    b_inds = np.arange(num_states)
    with tqdm.trange(cfg.epochs, desc = "Pretraining", colour = "green") as epochs:
        for _ in epochs:
            np.random.shuffle(b_inds)
            mean_kl = []
            for start in range(0, num_states, cfg.minibatch_size):
                mb_inds = b_inds[start:start+cfg.minibatch_size]

                optimizer.zero_grad()
                loss = loss_fn(agent.policy_linear, states, target_logprobs, mb_inds)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy_linear.parameters(), cfg.max_grad_norm)
                optimizer.step()
                mean_kl.append(loss.item())
            epochs.set_description(f"Pretraining | kl: {np.mean(mean_kl):.4f}")

    # Evaluate
    with torch.no_grad():
        total_correct = []
        for start in range(0, num_states, cfg.minibatch_size):
            mb_inds = np.arange(start, min(start + cfg.minibatch_size, num_states))
            _, logprobs, *_ = agent.policy_linear.get_action(
                {"mask": states.masks[mb_inds], "irreducible_edges": _sel(states.irreducible_edges, mb_inds)},
                valid_actions = states.valid_actions[mb_inds])
            total_correct.append(
                logprobs.argmax(dim = -1) == target_logprobs[mb_inds].argmax(dim = -1))
        print(f"Agreement: {torch.cat(total_correct).float().mean():.3f}")
