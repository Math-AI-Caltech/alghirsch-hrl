r"""train_ppoc.py
"""
from dataclasses import dataclass
from typing import Tuple
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
from models.agents import (
    Agent,
    Actor,
    QNetwork,
    MinQNetwork
)
from models.sgmodel import SGModel

from environments.alghirsch import AlgHirschEnv
from environments.options import ChainOption
from environments.wrappers import (
    TrajectoryRecorder,
    EpisodeStatistics,
    Autoreset
)

from structural.syzygies.sgraph import SGraph
from rl.pretrain import pretrain_linear, PretrainConfig
import rl.ppoc as ppoc

@dataclass
class Config:
    n: int = 7
    d: int = 4
    num_envs: int = 16

    device: str = "cuda"
    # PPO Config
    num_iterations: int = 100
    num_steps: int = 128
    lr: float = 2.5e-4

    gae_lambda: float = 0.95
    gamma: float = 0.99

    max_grad_norm: float = 0.5
    vf_coef: float  = 0.5
    ent_coef: float = 1e-2
    clip_vloss: bool = True
    clip_coef: float = 0.2
    norm_adv: bool = True

    minibatch_size: int = 64
    update_epochs: int = 4

    # Model config
    num_conv: int = 5
    embd_dim: int = 64

def initialize_env(cfg: Config):
    envs = AlgHirschEnv(
        num_envs    = cfg.num_envs,
        max_len     = cfg.d + 3 + 10,
        n           = cfg.n,
        d           = cfg.d,
        gamma       = cfg.gamma,
        device      = cfg.device)
    envs = ChainOption(envs)
    envs = EpisodeStatistics(envs)
    envs = TrajectoryRecorder(envs)
    envs = Autoreset(envs)
    return envs

def initialize_actor(sgraph: SGraph, cfg) -> Actor:
    return Actor(
        SGModel(
            sgraph = sgraph,
            num_conv = cfg.num_conv,
            node_embd_dim = cfg.embd_dim).to(cfg.device)).to(cfg.device)

def initialize_models(sgraph: SGraph, cfg) -> Tuple[Actor, MinQNetwork]:
    actor = initialize_actor(sgraph, cfg)

    q_net_1, q_net_2 = [QNetwork(
        SGModel(
            sgraph = sgraph,
            num_conv = cfg.num_conv,
            node_embd_dim = cfg.embd_dim).to(cfg.device)).to(cfg.device)
                for _ in range(2)]

    q_net = MinQNetwork(
        q_net_1 = q_net_1,
        q_net_2 = q_net_2)

    return actor, q_net

def train(cfg: Config, pretrain_cfg: PretrainConfig):
    envs = initialize_env(cfg)
    policy_spine, q_net_spine = initialize_models(
        sgraph  = envs._sgraph,
        cfg     = cfg)
    policy_linear, q_net_linear = initialize_models(
        sgraph  = envs._sgraph,
        cfg     = cfg)

    agent = Agent(
        policy_spine = policy_spine,
        q_net_spine = q_net_spine,
        policy_spine_optimizer = optim.Adam(
            policy_spine.parameters(), lr = cfg.lr, eps = .5e-4),
        q_spine_optimizer = optim.Adam(
            q_net_spine.parameters(), lr = cfg.lr, eps = .5e-4),

        policy_linear = policy_linear,
        q_net_linear = q_net_linear,
        policy_linear_optimizer = optim.Adam(
            policy_linear.parameters(), lr = cfg.lr, eps = .5e-4),
        q_linear_optimizer = optim.Adam(
            q_net_linear.parameters(), lr = cfg.lr, eps = .5e-4))

    pretrain_linear(envs, agent, pretrain_cfg)

    ret = envs.reset()
    batch_size = cfg.num_steps * envs.num_envs
    with tqdm.trange(1, cfg.num_iterations + 1) as iterations:
        for _ in iterations:
            batch, info = ppoc.explore_env(envs, agent, cfg, initial_state = ret)

            b_inds = np.arange(batch_size)
            for epoch in tqdm.trange(cfg.update_epochs, colour = "blue", desc = "Optimization"):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    agent.policy_spine_optimizer.zero_grad()
                    agent.q_spine_optimizer.zero_grad()
                    agent.policy_linear_optimizer.zero_grad()
                    agent.q_linear_optimizer.zero_grad()

                    loss_dict = ppoc.loss_fn(batch, mb_inds, agent, cfg)
                    loss = loss_dict["policy_loss"] - cfg.ent_coef * loss_dict["entropy_loss"] + cfg.vf_coef * loss_dict["value_loss"]
                    loss.backward()

                    nn.utils.clip_grad_norm_(agent.policy_spine.parameters(), cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent.q_net_spine.parameters(), cfg.max_grad_norm)

                    nn.utils.clip_grad_norm_(agent.policy_linear.parameters(), cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent.q_net_linear.parameters(), cfg.max_grad_norm)

                    agent.policy_spine_optimizer.step()
                    agent.q_spine_optimizer.step()
                    agent.policy_linear_optimizer.step()
                    agent.q_linear_optimizer.step()

            iterations.set_description(
                    f"#sols: {len(envs.terminal_trajectories)} | " + \
                    f"mean_return: {((info["total_return"] * info["total_episode_len"]).sum() / info["total_episode_len"].sum()).cpu().numpy()}")

if __name__ == "__main__":
    cfg = Config()
    pretrain_cfg = PretrainConfig()
    train(cfg, pretrain_cfg)