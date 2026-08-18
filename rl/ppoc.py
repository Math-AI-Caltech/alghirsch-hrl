r"""ppoc.py
"""
from typing import (
    List,
    Dict
)
from dataclasses import dataclass
import tqdm
import numpy as np

from functools import reduce
import torch

from models.agents import (
    Agent,
    get_action
)

from environments.alghirsch import AlgHirschEnv

def _sel(arr,ids): # type: ignore
    return [arr[i] for i in ids]

@dataclass
class Batch:
    b_obs: torch.Tensor
    b_irreducible_edges: List
    b_logprobs: torch.Tensor
    b_actions: torch.Tensor
    b_advantages: torch.Tensor
    b_returns: torch.Tensor
    b_values: torch.Tensor
    b_valid_actions: torch.Tensor
    b_dones: torch.Tensor
    b_options: torch.Tensor

def loss_fn(batch: Batch, mb_inds: np.ndarray, agent: Agent, cfg) -> Dict[str, torch.Tensor]:
    actions_data = get_action(
        agent = agent,
        x = {"mask": batch.b_obs[mb_inds], "irreducible_edges": _sel(batch.b_irreducible_edges, mb_inds)},
        valid_actions = batch.b_valid_actions[mb_inds],
        options = batch.b_options[mb_inds])
    new_logprob, entropy = actions_data["logprobs"], actions_data["entropy"]
    new_logprob = torch.gather(new_logprob, dim = 1, index = batch.b_actions[mb_inds].long().unsqueeze(dim=1)).squeeze()
    new_value = actions_data["values"]

    # Policy
    logratio = new_logprob - batch.b_logprobs[mb_inds]
    ratio = logratio.exp()

    minibatch_advantages = batch.b_advantages[mb_inds]
    if cfg.norm_adv:
        minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / minibatch_advantages.std()

    pg_loss1 = -minibatch_advantages * ratio
    pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value
    if cfg.clip_vloss:
        v_loss_unclipped = (new_value - batch.b_returns[mb_inds]) ** 2
        v_clipped = batch.b_values[mb_inds] + torch.clamp(
            new_value - batch.b_values[mb_inds],
            -cfg.clip_coef,
            cfg.clip_coef)
        v_loss_clipped = (v_clipped -  batch.b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((new_value - batch.b_returns[mb_inds]) ** 2).mean()

    return {
        "policy_loss": pg_loss,
        "value_loss": v_loss,
        "entropy_loss": entropy.mean()
    }

def explore_env(envs: AlgHirschEnv, agent: Agent, cfg, initial_state: Dict) -> Batch:
    ret = initial_state
    next_obs, info, next_irreducible_edges = ret["obs"], ret["info"], ret["irreducible_edges"]
    next_done = torch.zeros(envs.num_envs).to(cfg.device)

    obs_buffer = torch.zeros((cfg.num_steps, cfg.num_envs, envs.num_actions)).to(cfg.device)
    actions_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(cfg.device)
    logprobs_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(cfg.device)
    rewards_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(cfg.device)
    dones_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(cfg.device)
    values_buffer = torch.zeros((cfg.num_steps, cfg.num_envs)).to(cfg.device)
    valid_actions_buffer = torch.zeros((cfg.num_steps, cfg.num_envs, envs.num_actions)).to(cfg.device) == 1
    options_buffer = torch.zeros((cfg.num_steps, cfg.num_envs), dtype = torch.bool, device = cfg.device)
    irreducible_edges_buffer = [[] for _ in range(cfg.num_steps)]

    for step in tqdm.trange(cfg.num_steps, desc = "Explore"):
        obs_buffer[step] = next_obs
        irreducible_edges_buffer[step] = next_irreducible_edges
        dones_buffer[step] = next_done
        valid_actions_buffer[step] = ret["valid_actions"]
        options_buffer[step] = ret["options"]

        with torch.no_grad():
            actions_data = get_action(
                agent = agent,
                x = {"mask": obs_buffer[step], "irreducible_edges": irreducible_edges_buffer[step]},
                valid_actions = valid_actions_buffer[step],
                options = options_buffer[step])
            actions, logprobs, values = actions_data["actions"], actions_data["logprobs"], actions_data["values"]

        values_buffer[step] = values
        actions_buffer[step] = actions
        logprobs_buffer[step] = torch.gather(
            logprobs, dim = 1, index = actions.unsqueeze(dim=1)).squeeze()

        ret = envs.step(actions)
        next_obs, info, next_irreducible_edges = ret["obs"], ret["info"], ret["irreducible_edges"]
        next_done = torch.logical_or(ret["terminated"], ret["truncated"])
        rewards_buffer[step] = ret["reward"]

    # Estimate returns/advantages
    with torch.no_grad():
        actions_data = get_action(
            agent = agent,
            x = {"mask": next_obs, "irreducible_edges": next_irreducible_edges},
            valid_actions = ret["valid_actions"],
            options = ret["options"])
        logprobs, next_value = actions_data["logprobs"], actions_data["values"]

        advantages_buffer = torch.zeros_like(rewards_buffer)
        lastgaelam = 0
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nextnonterminal = 1.0 - next_done.type(torch.int64)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buffer[t + 1]
                nextvalues = values_buffer[t + 1]
            delta = rewards_buffer[t] + cfg.gamma * nextvalues * nextnonterminal - values_buffer[t]
            advantages_buffer[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
        returns_buffer = advantages_buffer + values_buffer

    return Batch(
        b_obs = obs_buffer.reshape((-1, obs_buffer.shape[-1])),
        b_irreducible_edges = reduce(lambda x,y:x+y, irreducible_edges_buffer),
        b_logprobs = logprobs_buffer.reshape(-1),
        b_actions = actions_buffer.reshape(-1),
        b_advantages = advantages_buffer.reshape(-1),
        b_returns = returns_buffer.reshape(-1),
        b_values = values_buffer.reshape(-1),
        b_valid_actions = valid_actions_buffer.reshape(-1, envs.num_actions),
        b_dones = dones_buffer.reshape(-1),
        b_options = options_buffer.reshape(-1)), info
